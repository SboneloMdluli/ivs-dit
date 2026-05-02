"""Preset Heston-IV-surface YAML overlays."""

from enum import Enum


class HestonIvGoal(str, Enum):
    """Preset Heston IV-surface YAML overlay; value is the stem of the overlay file."""

    LOW_VOL = "low_vol"
    HIGH_VOL = "high_vol"
    SKEW = "skew"
    SMILE = "smile"
    SEQUENTIAL_PATH = "sequential_path"


def heston_goal_overlay_filename(goal: HestonIvGoal) -> str:
    """Config filename for ``goal`` (under the config directory)."""
    return f"heston_goal_{goal.value}.yaml"


HESTON_GOAL_YAML: dict[HestonIvGoal, str] = {goal: heston_goal_overlay_filename(goal) for goal in HestonIvGoal}


def coerce_heston_iv_goal(goal: HestonIvGoal | str) -> HestonIvGoal:
    """Accept either the enum member or its ``value`` string."""
    if isinstance(goal, HestonIvGoal):
        return goal
    key = str(goal).strip()
    try:
        return HestonIvGoal(key)
    except ValueError:
        allowed = ", ".join(sorted(x.value for x in HestonIvGoal))
        raise ValueError(f"unknown goal {goal!r}; expected one of: {allowed}") from None
