"""Common result types returned by robot actions."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SkillResult:
    ok: bool
    status: str


@dataclass(frozen=True)
class FixedPlaceResult(SkillResult):
    pass


@dataclass(frozen=True)
class RobotActionResult:
    status: str
    failed_min_limit: bool = False
    failed_hand: str | None = None
    grasp_confirmed: bool = False
    grasp_hand: str | None = None
    object_label: str | None = None
    ok: bool = False
