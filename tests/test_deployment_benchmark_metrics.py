import pytest

from benchmarks.deployment.metrics import growth_summary, latency_summary, parse_gpu_row


def test_latency_summary_uses_interpolated_percentiles() -> None:
    result = latency_summary([1.0, 2.0, 3.0, 4.0])
    assert result["average_ms"] == 2.5
    assert result["p50_ms"] == 2.5
    assert result["p95_ms"] == pytest.approx(3.85)
    assert result["max_ms"] == 4.0


def test_growth_summary_flags_sustained_growth() -> None:
    result = growth_summary([float(index * 10) for index in range(100)], 64.0)
    assert result["delta_mib"] > 64.0
    assert result["slope_mib_per_iteration"] == 10.0
    assert result["suspected_leak"] is True


def test_growth_summary_ignores_allocator_plateau() -> None:
    result = growth_summary([100.0, 164.0] + [164.0] * 98, 64.0)
    assert result["suspected_leak"] is False


def test_parse_gpu_row() -> None:
    assert parse_gpu_row("0, 87, 4096, 8192, 71") == {
        "index": 0,
        "util_percent": 87.0,
        "used_mib": 4096.0,
        "total_mib": 8192.0,
        "temperature_c": 71.0,
    }
