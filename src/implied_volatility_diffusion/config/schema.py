"""Lightweight, optional validation around the synthetic-surface config dicts.

The rest of the library accepts plain ``Mapping`` objects; ``SurfaceRecipe``
is a convenience for users who want field-level errors up front instead of
``KeyError``s deep inside a pricing call.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class SurfaceRecipe:
    """Typed view over a surface-generation config ``dict``.

    Attributes:
        market: Required ``spot``; optional ``dividend_yield`` / ``rate``.
        grid: Moneyness + tau axis specs used by :func:`grid_axes`.
        lhs: Sampling settings (``n_samples``, ``seed``, ...).
        ranges: Parameter box ranges per model parameter.
        extra: All remaining keys, passed through untouched (e.g. ``heston_cos_pricer``).
    """

    market: Mapping[str, Any]
    grid: Mapping[str, Any]
    lhs: Mapping[str, Any]
    ranges: Mapping[str, Any]
    extra: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, cfg: Mapping[str, Any], *, ranges_key: str) -> "SurfaceRecipe":
        """Validate and pick the known keys out of ``cfg``.

        ``ranges_key`` selects e.g. ``"heston_ranges"`` or ``"sabr_ranges"``.
        """
        if "market" not in cfg or "spot" not in cfg["market"]:
            raise KeyError("config requires market.spot")
        if "grid" not in cfg:
            raise KeyError("config requires grid section")
        if ranges_key not in cfg:
            raise KeyError(f"config requires {ranges_key!r} section")
        known = {"market", "grid", "lhs", ranges_key}
        extra = {k: v for k, v in cfg.items() if k not in known}
        return cls(
            market=dict(cfg["market"]),
            grid=dict(cfg["grid"]),
            lhs=dict(cfg.get("lhs", {})),
            ranges=dict(cfg[ranges_key]),
            extra=extra,
        )

    def to_dict(self, *, ranges_key: str) -> dict[str, Any]:
        """Re-materialize a plain dict that the legacy helpers accept."""
        out: dict[str, Any] = {
            "market": dict(self.market),
            "grid": dict(self.grid),
            "lhs": dict(self.lhs),
            ranges_key: dict(self.ranges),
        }
        out.update(self.extra)
        return out
