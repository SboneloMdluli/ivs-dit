"""Arbitrage penalties Φ(σ) and location matrices P1, P2, P3 (Cont & Vuletić)."""

from dataclasses import dataclass

import numpy as np

from implied_volatility_diffusion.arbitrage_checks.checks import _bs_call_grid


@dataclass(frozen=True)
class SurfaceArbitrageWeights:
    """Per-family multipliers; set to 0 to disable a family."""

    calendar: float = 1.0
    butterfly: float = 1.0
    call: float = 1.0


@dataclass(frozen=True)
class PenaltyMatrices:
    """Per-grid-point violation magnitudes for calendar, call, and butterfly constraints."""

    P1_calendar: np.ndarray
    P2_call: np.ndarray
    P3_butterfly: np.ndarray

    @property
    def total(self) -> np.ndarray:
        """Combined penalty matrix P1 + P2 + P3."""
        return self.P1_calendar + self.P2_call + self.P3_butterfly

    @property
    def phi(self) -> float:
        """Scalar penalty Φ(σ) = ||P1 + P2 + P3||_1 (sum of all entries)."""
        return float(np.sum(self.total))

    @property
    def phi_calendar(self) -> float:
        return float(np.sum(self.P1_calendar))

    @property
    def phi_call(self) -> float:
        return float(np.sum(self.P2_call))

    @property
    def phi_butterfly(self) -> float:
        return float(np.sum(self.P3_butterfly))

    @property
    def arbitrage_free(self) -> bool:
        return self.phi == 0.0


def smoothness_penalty_moneyness(log_iv: np.ndarray, moneyness: np.ndarray) -> float:
    """VolGAN L_m: squared finite differences of log-IV along moneyness."""
    g = np.asarray(log_iv, dtype=float)
    m = np.asarray(moneyness, dtype=float).reshape(-1)
    if m.size < 2:
        return 0.0
    dm = np.diff(m).reshape(-1, 1)
    dg = np.diff(g, axis=0)
    return float(np.sum((dg / dm) ** 2))


def smoothness_penalty_tau(log_iv: np.ndarray, tau: np.ndarray) -> float:
    """VolGAN L_τ: squared finite differences of log-IV along tau."""
    g = np.asarray(log_iv, dtype=float)
    t = np.asarray(tau, dtype=float).reshape(-1)
    if t.size < 2:
        return 0.0
    dt = np.diff(t).reshape(1, -1)
    dg = np.diff(g, axis=1)
    return float(np.sum((dg / dt) ** 2))


def smoothness_penalty(
    iv: np.ndarray,
    moneyness: np.ndarray,
    tau: np.ndarray,
    *,
    iv_floor: float = 1e-4,
) -> tuple[float, float]:
    """Return ``(L_m, L_tau)`` smoothness penalties on log(IV)."""
    iv_arr = np.asarray(iv, dtype=float)
    valid = np.isfinite(iv_arr) & (iv_arr > iv_floor)
    log_iv = np.where(valid, np.log(np.maximum(iv_arr, iv_floor)), 0.0)
    lm = smoothness_penalty_moneyness(log_iv, moneyness)
    lt = smoothness_penalty_tau(log_iv, tau)
    return lm, lt


@dataclass(frozen=True)
class SurfaceArbitragePenalty:
    """Scalar Φ(σ) and penalty matrices for an IV grid."""

    moneyness: np.ndarray
    tau: np.ndarray
    spot: float = 1.0
    rate: float = 0.0
    dividend_yield: float = 0.0
    weights: SurfaceArbitrageWeights | None = None

    def __post_init__(self) -> None:
        m = np.asarray(self.moneyness, dtype=float).reshape(-1)
        t = np.asarray(self.tau, dtype=float).reshape(-1)
        if m.size < 1 or t.size < 1:
            raise ValueError("moneyness and tau must be non-empty")
        if bool(np.any(np.diff(m) <= 0.0)):
            raise ValueError("moneyness must be strictly increasing")
        if bool(np.any(np.diff(t) <= 0.0)):
            raise ValueError("tau must be strictly increasing")
        object.__setattr__(self, "moneyness", m)
        object.__setattr__(self, "tau", t)
        object.__setattr__(self, "strikes", m * float(self.spot))

    @property
    def grid_shape(self) -> tuple[int, int]:
        return int(self.moneyness.size), int(self.tau.size)

    def _ensure_iv(self, iv: np.ndarray) -> np.ndarray:
        iv_arr = np.asarray(iv, dtype=float)
        if iv_arr.shape[-2:] != self.grid_shape:
            raise ValueError(f"IV trailing shape {iv_arr.shape[-2:]} must match grid {self.grid_shape}")
        return iv_arr

    def call_prices(self, iv: np.ndarray) -> np.ndarray:
        iv_arr = self._ensure_iv(iv)
        return _bs_call_grid(
            iv_arr,
            self.moneyness,
            self.tau,
            spot=self.spot,
            rate=self.rate,
            dividend_yield=self.dividend_yield,
        )

    def penalty_matrices(self, iv: np.ndarray) -> PenaltyMatrices:
        """Location-specific P1 (calendar), P2 (call), P3 (butterfly) violation grids."""
        iv_arr = self._ensure_iv(iv)
        n_m, n_t = self.grid_shape
        c = self.call_prices(iv_arr)
        k = self.strikes
        t = self.tau

        P1 = np.zeros((n_m, n_t), dtype=float)
        if n_t >= 2:
            dt = np.diff(t).reshape(1, -1)
            dc_dt = np.diff(c, axis=1) / dt
            cal_viol = np.maximum(0.0, -(t[:-1].reshape(1, -1) * dc_dt))
            P1[:, :-1] = np.where(np.isfinite(cal_viol), cal_viol, 0.0)

        P2 = np.zeros((n_m, n_t), dtype=float)
        if n_m >= 2:
            dk = np.diff(k).reshape(-1, 1)
            slope = np.diff(c, axis=0) / dk
            call_viol = np.maximum(0.0, slope)
            P2[:-1, :] = np.where(np.isfinite(call_viol), call_viol, 0.0)

        P3 = np.zeros((n_m, n_t), dtype=float)
        if n_m >= 3:
            dk_l = (k[1:-1] - k[:-2]).reshape(-1, 1)
            dk_r = (k[2:] - k[1:-1]).reshape(-1, 1)
            denom = dk_l * dk_r * (dk_l + dk_r) / 2.0
            d2 = (dk_l * c[2:, :] - (dk_l + dk_r) * c[1:-1, :] + dk_r * c[:-2, :]) / denom
            bfly_viol = np.maximum(0.0, -d2)
            P3[1:-1, :] = np.where(np.isfinite(bfly_viol), bfly_viol, 0.0)

        return PenaltyMatrices(P1_calendar=P1, P2_call=P2, P3_butterfly=P3)

    def calendar_penalty(self, iv: np.ndarray) -> float:
        iv_arr = self._ensure_iv(iv)
        if iv_arr.shape[-1] < 2:
            return 0.0
        w = (iv_arr * iv_arr) * self.tau.reshape(1, -1)
        diff = np.diff(w, axis=-1)
        return float(np.mean(np.maximum(0.0, -diff)))

    def butterfly_penalty(self, call_prices: np.ndarray) -> float:
        c = np.asarray(call_prices, dtype=float)
        k = self.strikes
        if k.size < 3:
            return 0.0
        dk_l = (k[1:-1] - k[:-2]).reshape(-1, 1)
        dk_r = (k[2:] - k[1:-1]).reshape(-1, 1)
        denom = dk_l * dk_r * (dk_l + dk_r) / 2.0
        d2 = (dk_l * c[2:, :] - (dk_l + dk_r) * c[1:-1, :] + dk_r * c[:-2, :]) / denom
        return float(np.mean(np.maximum(0.0, -d2)))

    def call_penalty(self, call_prices: np.ndarray) -> float:
        c = np.asarray(call_prices, dtype=float)
        k = self.strikes
        if k.size < 2:
            return 0.0
        dk = (k[1:] - k[:-1]).reshape(-1, 1)
        slope = np.diff(c, axis=0) / dk
        return float(np.mean(np.maximum(0.0, slope)))

    def smoothness(self, iv: np.ndarray, *, iv_floor: float = 1e-4) -> tuple[float, float]:
        """VolGAN smoothness penalties (L_m, L_τ) on the log-IV surface."""
        iv_arr = self._ensure_iv(iv)
        return smoothness_penalty(iv_arr, self.moneyness, self.tau, iv_floor=iv_floor)

    def forward(self, iv: np.ndarray) -> dict[str, float]:
        w = self.weights or SurfaceArbitrageWeights()
        iv_arr = self._ensure_iv(iv)
        need_calls = w.butterfly > 0.0 or w.call > 0.0
        c = self.call_prices(iv_arr) if need_calls else np.zeros(iv_arr.shape)
        return {
            "calendar": self.calendar_penalty(iv_arr) * w.calendar,
            "butterfly": self.butterfly_penalty(c) * w.butterfly,
            "call": self.call_penalty(c) * w.call,
        }

    def __call__(self, iv: np.ndarray) -> float:
        """Total penalty Φ(σ); zero when the surface is arbitrage-free."""
        parts = self.forward(iv)
        return float(sum(parts.values()))

    def batch(self, iv: np.ndarray) -> np.ndarray:
        """Per-scenario penalties for ``(..., M, T)`` IV batches."""
        iv_arr = np.asarray(iv, dtype=float)
        leading = iv_arr.shape[:-2]
        flat = iv_arr.reshape(-1, *self.grid_shape)
        out = np.array([self(s) for s in flat], dtype=float)
        return out.reshape(leading) if leading else out.reshape(())
