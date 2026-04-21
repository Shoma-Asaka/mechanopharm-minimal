from __future__ import annotations

import numpy as np


def ec50_from_curve(c: np.ndarray, y: np.ndarray) -> float:
    """Estimate the half-maximal concentration from a monotonic dose-response curve.

    The concentration grid must be one-dimensional and strictly increasing.
    The response is assumed to be monotonic over the sampled concentration range;
    non-monotonic response landscapes should be summarized with a different metric.
    """
    c = np.asarray(c, dtype=float)
    y = np.asarray(y, dtype=float)
    if c.ndim != 1 or y.ndim != 1 or len(c) != len(y):
        raise ValueError("c and y must be 1D arrays with the same length")
    if len(c) < 2:
        raise ValueError("c and y must contain at least two points")
    if not np.all(np.isfinite(c)) or not np.all(np.isfinite(y)):
        raise ValueError("c and y must contain only finite values")
    if np.any(np.diff(c) <= 0):
        raise ValueError("c must be strictly increasing")
    ymin = float(np.min(y))
    ymax = float(np.max(y))
    target = ymin + 0.5 * (ymax - ymin)
    increasing = y[-1] >= y[0]
    if not increasing:
        y = y[::-1]
        c = c[::-1]
    if target < np.min(y) or target > np.max(y):
        return float("nan")
    return float(np.interp(target, y, c))


def ec50_vs_m(c_grid: np.ndarray, m_grid: np.ndarray, response: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute EC50 estimates for each mechanical condition.

    ``response`` must have shape ``(len(m_grid), len(c_grid))``.
    """
    c_grid = np.asarray(c_grid, dtype=float)
    m_grid = np.asarray(m_grid, dtype=float)
    response = np.asarray(response, dtype=float)
    if response.shape != (len(m_grid), len(c_grid)):
        raise ValueError("response must have shape (len(m_grid), len(c_grid))")
    ec50 = np.array([ec50_from_curve(c_grid, response[i, :]) for i in range(len(m_grid))])
    return m_grid, ec50


def find_mechanical_optima(c_grid: np.ndarray, m_grid: np.ndarray, response: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Find the response-maximizing mechanical condition at each concentration.

    ``response`` must have shape ``(len(m_grid), len(c_grid))``.
    """
    c_grid = np.asarray(c_grid, dtype=float)
    m_grid = np.asarray(m_grid, dtype=float)
    response = np.asarray(response, dtype=float)
    if response.shape != (len(m_grid), len(c_grid)):
        raise ValueError("response must have shape (len(m_grid), len(c_grid))")
    idx = np.argmax(response, axis=0)
    m_opt = m_grid[idx]
    return c_grid, m_opt


def mechanical_sign_reversal(c_grid: np.ndarray, m_grid: np.ndarray, response: np.ndarray) -> dict[str, float | bool]:
    """
    Very simple diagnostic:
    compare average slope wrt m at low c and high c.
    """
    c_grid = np.asarray(c_grid, dtype=float)
    m_grid = np.asarray(m_grid, dtype=float)
    response = np.asarray(response, dtype=float)

    dm = np.gradient(m_grid)
    dRdm = np.gradient(response, axis=0) / dm[:, None]
    low = np.nanmean(dRdm[:, : max(2, len(c_grid)//4)])
    high = np.nanmean(dRdm[:, -max(2, len(c_grid)//4):])
    has_reversal = np.sign(low) != np.sign(high) and abs(low) > 1e-8 and abs(high) > 1e-8

    return {
        "low_c_mean_slope": float(low),
        "high_c_mean_slope": float(high),
        "has_reversal": bool(has_reversal),
    }


def peak_metrics_by_condition(time: np.ndarray, values_by_condition: dict[tuple[float, float], np.ndarray]) -> list[dict[str, float]]:
    time = np.asarray(time, dtype=float)
    out: list[dict[str, float]] = []
    for (c, m), y in values_by_condition.items():
        y = np.asarray(y, dtype=float)
        idx = int(np.argmax(y))
        out.append(
            {
                "c": float(c),
                "m": float(m),
                "peak_value": float(y[idx]),
                "peak_time": float(time[idx]),
            }
        )
    return out
