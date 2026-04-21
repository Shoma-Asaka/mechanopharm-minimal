from __future__ import annotations

from dataclasses import dataclass
import numpy as np


ArrayLike = np.ndarray


@dataclass
class TwoStateModel:
    """
    Minimal thermodynamically constrained two-state model.

    Parameters map directly to the paper's reduced bias:
        ΔG(c,m) = ΔG0 - Δα c - Δλ m - Δμ c m
    """
    beta: float = 1.0
    delta_g0: float = 1.2
    delta_alpha: float = 2.0
    delta_lambda: float = 0.8
    delta_mu: float = -0.9

    def delta_g(self, c: ArrayLike, m: ArrayLike) -> ArrayLike:
        c = np.asarray(c)
        m = np.asarray(m)
        return (
            self.delta_g0
            - self.delta_alpha * c
            - self.delta_lambda * m
            - self.delta_mu * c * m
        )

    def occupancy(self, c: ArrayLike, m: ArrayLike) -> ArrayLike:
        dg = self.delta_g(c, m)
        return 1.0 / (1.0 + np.exp(self.beta * dg))

    def signal(self, c: ArrayLike, m: ArrayLike, eps0: float = 0.0, eps1: float = 1.0) -> ArrayLike:
        p = self.occupancy(c, m)
        return eps0 * (1.0 - p) + eps1 * p

    def c_half(self, m: ArrayLike) -> ArrayLike:
        m_arr = np.asarray(m, dtype=float)
        denom = self.delta_alpha + self.delta_mu * m_arr
        out = np.full_like(m_arr, np.nan, dtype=float)
        mask = np.abs(denom) > 1e-12
        out[mask] = (self.delta_g0 - self.delta_lambda * m_arr[mask]) / denom[mask]
        if np.ndim(m) == 0:
            return float(out)
        return out

    def mechanical_sensitivity_prefactor(self, c: ArrayLike) -> ArrayLike:
        c = np.asarray(c)
        return self.delta_lambda + self.delta_mu * c

    def reversal_concentration(self) -> float | None:
        if abs(self.delta_mu) < 1e-12:
            return None
        return -self.delta_lambda / self.delta_mu


@dataclass
class ThreeStateProtectionModel:
    """
    Minimal phenomenological three-state model for adaptive protection.

    This is intentionally simple:
    - state 1 is the responsive state,
    - state 2 is a protected macrostate,
    - p1(c,m) shows an interior optimum in m,
    - time courses show transient sensitivity followed by delayed protection.

    The current implementation is a compact reference model for experimentation,
    not a unique mechanistic fit to any biological platform.
    """
    mopt_intercept: float = 0.30
    mopt_slope: float = 0.48
    width_base: float = 0.22
    width_slope: float = 0.02
    baseline_base: float = 0.08
    baseline_scale: float = 0.04
    amp_base: float = 0.58
    amp_scale: float = 0.26

    def optimal_load(self, c: ArrayLike) -> ArrayLike:
        c = np.asarray(c)
        return self.mopt_intercept + self.mopt_slope * c

    def responsive_fraction_steady(self, c: ArrayLike, m: ArrayLike) -> ArrayLike:
        c = np.asarray(c)
        m = np.asarray(m)
        mopt = self.optimal_load(c)
        amp = self.amp_base + self.amp_scale * np.tanh(1.3 * (c - 0.55))
        width = self.width_base + self.width_slope * c
        baseline = self.baseline_base + self.baseline_scale * np.tanh(1.0 * (c - 0.7))
        return baseline + amp * np.exp(-((m - mopt) ** 2) / (2.0 * width ** 2))

    def responsive_fraction_timecourse(self, t: ArrayLike, c: float, m: float) -> ArrayLike:
        t = np.asarray(t)
        amp = 0.28 + 0.30 * np.exp(-((m - self.optimal_load(c)) ** 2) / (2 * 0.22 ** 2))
        tau_fast = 1.0 + 0.45 * m + 0.18 / (c + 0.15)
        tau_escape = max(5.2 - 1.7 * m + 0.55 * c, 1.3)
        return amp * (1.0 - np.exp(-t / tau_fast)) * np.exp(-t / tau_escape)

    def protected_fraction_timecourse(self, t: ArrayLike, c: float, m: float) -> ArrayLike:
        t = np.asarray(t)
        amp2 = 0.18 + 0.36 / (1.0 + np.exp(-4.0 * (m - 0.7)))
        tau2 = max(5.5 - 1.2 * m + 0.6 * c, 1.5)
        return amp2 * (1.0 - np.exp(-t / tau2))

    def baseline_fraction_timecourse(self, t: ArrayLike, c: float, m: float) -> ArrayLike:
        p1 = self.responsive_fraction_timecourse(t, c, m)
        p2 = self.protected_fraction_timecourse(t, c, m)
        p0 = 1.0 - p1 - p2
        return np.clip(p0, 0.0, 1.0)

    def peak_metrics(self, c: float, m: float, t_max: float = 20.0, n_t: int = 1200) -> tuple[float, float]:
        t = np.linspace(0.0, t_max, n_t)
        y = self.responsive_fraction_timecourse(t, c, m)
        idx = int(np.argmax(y))
        return float(y[idx]), float(t[idx])


def rk4_step(fun, t: float, y: np.ndarray, dt: float, *args) -> np.ndarray:
    k1 = fun(t, y, *args)
    k2 = fun(t + 0.5 * dt, y + 0.5 * dt * k1, *args)
    k3 = fun(t + 0.5 * dt, y + 0.5 * dt * k2, *args)
    k4 = fun(t + dt, y + dt * k3, *args)
    return y + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


def simulate_three_state_timecourse(
    t: ArrayLike,
    c: float,
    m: float,
    p0_init: float = 1.0,
    p1_init: float = 0.0,
    p2_init: float = 0.0,
) -> dict[str, np.ndarray]:
    """
    Optional explicit chain model:
        0 <-> 1 <-> 2
    with rate choices selected only to produce a clean minimal transient.

    This function is useful when one wants a true occupancy-conserving ODE system.
    """
    t = np.asarray(t, dtype=float)
    if t.ndim != 1 or len(t) < 2:
        raise ValueError("t must be a 1D array with at least two time points")
    if not np.all(np.isfinite(t)):
        raise ValueError("t must contain only finite values")
    if np.any(np.diff(t) <= 0):
        raise ValueError("t must be strictly increasing")

    y0 = np.array([p0_init, p1_init, p2_init], dtype=float)
    if not np.all(np.isfinite(y0)):
        raise ValueError("initial occupancies must be finite")
    if np.any(y0 < 0):
        raise ValueError("initial occupancies must be non-negative")
    if y0.sum() <= 0:
        raise ValueError("initial occupancies must have positive total mass")
    y0 = y0 / y0.sum()

    def rates(c_: float, m_: float) -> tuple[float, float, float, float]:
        k01 = 0.16 + 0.42 / (1.0 + np.exp(-(1.7 * c_ + 0.8 * m_ - 1.2)))
        k10 = 0.10 + 0.05 * m_
        k12 = 0.04 + 0.23 / (1.0 + np.exp(-(2.0 * m_ + 0.9 * c_ - 2.0)))
        k21 = 0.05 + 0.02 / (1.0 + c_)
        return k01, k10, k12, k21

    def rhs(_t: float, y: np.ndarray, c_: float, m_: float) -> np.ndarray:
        p0, p1, p2 = y
        k01, k10, k12, k21 = rates(c_, m_)
        dp0 = -k01 * p0 + k10 * p1
        dp1 = k01 * p0 - k10 * p1 - k12 * p1 + k21 * p2
        dp2 = k12 * p1 - k21 * p2
        return np.array([dp0, dp1, dp2])

    y = np.zeros((len(t), 3), dtype=float)
    y[0] = y0

    for i in range(1, len(t)):
        dt = float(t[i] - t[i - 1])
        y[i] = rk4_step(rhs, float(t[i - 1]), y[i - 1], dt, c, m)
        y[i] = np.clip(y[i], 0.0, 1.0)
        s = y[i].sum()
        if s > 0:
            y[i] /= s

    return {"t": t, "p0": y[:, 0], "p1": y[:, 1], "p2": y[:, 2]}
