"""Low-overhead host and NVIDIA GPU sampling used by the deployment benchmark."""

from __future__ import annotations

import os
import statistics
import subprocess
import threading
import time
from dataclasses import asdict, dataclass
from typing import Any, Iterable

import psutil


MIB = 1024 * 1024


@dataclass(frozen=True)
class Sample:
    monotonic_sec: float
    process_rss_mib: float
    process_cpu_percent: float
    system_cpu_percent: float
    system_available_mib: float
    system_used_mib: float
    gpu_used_mib: float | None
    gpu_total_mib: float | None
    gpu_util_percent: float | None
    process_gpu_mib: float | None


def _number(value: str) -> float | None:
    value = value.strip()
    if not value or value in {"N/A", "[N/A]", "Not Supported"}:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def parse_gpu_row(row: str) -> dict[str, float | int | None]:
    """Parse the CSV row produced by ``query-gpu`` (kept public for tests)."""
    fields = [part.strip() for part in row.split(",")]
    if len(fields) != 5:
        raise ValueError(f"unexpected nvidia-smi GPU row: {row!r}")
    return {
        "index": int(fields[0]),
        "util_percent": _number(fields[1]),
        "used_mib": _number(fields[2]),
        "total_mib": _number(fields[3]),
        "temperature_c": _number(fields[4]),
    }


class NvidiaSmi:
    def __init__(self, gpu_index: int = 0, pid: int | None = None) -> None:
        self.gpu_index = gpu_index
        self.pid = os.getpid() if pid is None else pid

    @staticmethod
    def _run(args: list[str]) -> str:
        completed = subprocess.run(
            ["nvidia-smi", *args],
            check=True,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
        return completed.stdout.strip()

    def identity(self) -> dict[str, Any]:
        try:
            output = self._run(
                [
                    f"--id={self.gpu_index}",
                    "--query-gpu=index,name,uuid,driver_version,memory.total,compute_cap",
                    "--format=csv,noheader,nounits",
                ]
            )
            fields = [part.strip() for part in output.splitlines()[0].split(",")]
            return {
                "index": int(fields[0]),
                "name": fields[1],
                "uuid": fields[2],
                "driver_version": fields[3],
                "memory_total_mib": _number(fields[4]),
                "compute_capability": fields[5],
            }
        except Exception as exc:  # nvidia-smi may be absent on CPU-only hosts.
            return {"index": self.gpu_index, "error": str(exc)}

    def sample(self) -> dict[str, float | None]:
        gpu: dict[str, float | int | None] = {}
        process_gpu_mib: float | None = None
        try:
            output = self._run(
                [
                    f"--id={self.gpu_index}",
                    "--query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu",
                    "--format=csv,noheader,nounits",
                ]
            )
            gpu = parse_gpu_row(output.splitlines()[0])
        except Exception:
            pass
        try:
            output = self._run(
                [
                    "--query-compute-apps=pid,used_memory",
                    "--format=csv,noheader,nounits",
                ]
            )
            matches = []
            for row in output.splitlines():
                fields = [part.strip() for part in row.split(",")]
                if len(fields) == 2 and fields[0].isdigit() and int(fields[0]) == self.pid:
                    value = _number(fields[1])
                    if value is not None:
                        matches.append(value)
            process_gpu_mib = sum(matches) if matches else 0.0
        except Exception:
            pass
        return {
            "gpu_used_mib": gpu.get("used_mib"),
            "gpu_total_mib": gpu.get("total_mib"),
            "gpu_util_percent": gpu.get("util_percent"),
            "process_gpu_mib": process_gpu_mib,
        }


def take_sample(process: psutil.Process, nvidia: NvidiaSmi) -> Sample:
    virtual = psutil.virtual_memory()
    gpu = nvidia.sample()
    return Sample(
        monotonic_sec=time.perf_counter(),
        process_rss_mib=process.memory_info().rss / MIB,
        process_cpu_percent=process.cpu_percent(None),
        system_cpu_percent=psutil.cpu_percent(None),
        system_available_mib=virtual.available / MIB,
        system_used_mib=virtual.used / MIB,
        **gpu,
    )


class ResourceSampler:
    def __init__(self, interval_sec: float, gpu_index: int = 0) -> None:
        self.interval_sec = max(0.05, interval_sec)
        self.process = psutil.Process()
        self.nvidia = NvidiaSmi(gpu_index=gpu_index)
        self.samples: list[Sample] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self.process.cpu_percent(None)
        psutil.cpu_percent(None)
        self.samples.append(take_sample(self.process, self.nvidia))
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.wait(self.interval_sec):
            self.samples.append(take_sample(self.process, self.nvidia))

    def checkpoint(self) -> dict[str, Any]:
        sample = take_sample(self.process, self.nvidia)
        self.samples.append(sample)
        return asdict(sample)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.interval_sec * 3))
        self.samples.append(take_sample(self.process, self.nvidia))

    def summary(self) -> dict[str, Any]:
        return summarize_samples(self.samples)


def _values(samples: Iterable[Sample], name: str) -> list[float]:
    return [float(value) for item in samples if (value := getattr(item, name)) is not None]


def summarize_samples(samples: list[Sample]) -> dict[str, Any]:
    def stats(name: str) -> dict[str, float | None]:
        values = _values(samples, name)
        if not values:
            return {"min": None, "average": None, "max": None}
        return {
            "min": min(values),
            "average": statistics.fmean(values),
            "max": max(values),
        }

    return {
        "sample_count": len(samples),
        "process_rss_mib": stats("process_rss_mib"),
        "process_cpu_percent": stats("process_cpu_percent"),
        "system_cpu_percent": stats("system_cpu_percent"),
        "system_available_mib": stats("system_available_mib"),
        "system_used_mib": stats("system_used_mib"),
        "gpu_used_mib": stats("gpu_used_mib"),
        "gpu_util_percent": stats("gpu_util_percent"),
        "process_gpu_mib": stats("process_gpu_mib"),
    }


def latency_summary(milliseconds: list[float]) -> dict[str, float | int | None]:
    if not milliseconds:
        return {"count": 0, "average_ms": None, "p50_ms": None, "p95_ms": None, "max_ms": None}
    ordered = sorted(milliseconds)

    def percentile(fraction: float) -> float:
        if len(ordered) == 1:
            return ordered[0]
        position = (len(ordered) - 1) * fraction
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        weight = position - lower
        return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

    return {
        "count": len(milliseconds),
        "average_ms": statistics.fmean(milliseconds),
        "p50_ms": percentile(0.50),
        "p95_ms": percentile(0.95),
        "max_ms": max(milliseconds),
    }


def growth_summary(values_mib: list[float], leak_threshold_mib: float) -> dict[str, Any]:
    """Compare steady-state head/tail windows and calculate linear growth."""
    if not values_mib:
        return {"count": 0, "delta_mib": None, "slope_mib_per_iteration": None, "suspected_leak": False}
    window = max(1, min(10, len(values_mib) // 4 or 1))
    head = statistics.median(values_mib[:window])
    tail = statistics.median(values_mib[-window:])
    count = len(values_mib)
    x_mean = (count - 1) / 2
    y_mean = statistics.fmean(values_mib)
    denominator = sum((index - x_mean) ** 2 for index in range(count))
    slope = (
        sum((index - x_mean) * (value - y_mean) for index, value in enumerate(values_mib)) / denominator
        if denominator
        else 0.0
    )
    delta = tail - head
    return {
        "count": count,
        "head_median_mib": head,
        "tail_median_mib": tail,
        "delta_mib": delta,
        "slope_mib_per_iteration": slope,
        "threshold_mib": leak_threshold_mib,
        "suspected_leak": bool(delta > leak_threshold_mib and slope > 0),
    }
