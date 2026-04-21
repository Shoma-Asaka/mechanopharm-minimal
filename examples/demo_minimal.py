from __future__ import annotations

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import matplotlib.pyplot as plt

from mechanopharm_minimal.models import (
    TwoStateModel,
    ThreeStateProtectionModel,
    simulate_three_state_timecourse,
)
from mechanopharm_minimal.fingerprints import (
    ec50_vs_m,
    find_mechanical_optima,
    mechanical_sign_reversal,
    peak_metrics_by_condition,
)
from mechanopharm_minimal.plotting import (
    plot_two_state_landscape,
    plot_three_state_timecourse,
)


OUTDIR = Path("outputs")
OUTDIR.mkdir(exist_ok=True)


def save_endpoint_csv(path: Path, c_grid: np.ndarray, m_grid: np.ndarray, response: np.ndarray) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("c,m,response\n")
        for i, m in enumerate(m_grid):
            for j, c in enumerate(c_grid):
                f.write(f"{c:.6f},{m:.6f},{response[i, j]:.6f}\n")


def save_timecourse_csv(path: Path, t: np.ndarray, traces: dict[tuple[float, float], np.ndarray]) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("time,c,m,value\n")
        for (c, m), y in traces.items():
            for ti, yi in zip(t, y):
                f.write(f"{ti:.6f},{c:.6f},{m:.6f},{yi:.6f}\n")


def main() -> None:
    two = TwoStateModel()
    c_grid = np.linspace(0.0, 2.0, 220)
    m_grid = np.linspace(-1.0, 1.0, 220)
    C, M = np.meshgrid(c_grid, m_grid)
    R2 = two.signal(C, M)

    fig, _ = plot_two_state_landscape(c_grid, m_grid, R2, savepath=str(OUTDIR / "two_state_landscape.png"))
    plt.close(fig)
    save_endpoint_csv(OUTDIR / "synthetic_endpoint.csv", c_grid, m_grid, R2)

    _, ec50_vals = ec50_vs_m(c_grid, m_grid, R2)
    reversal = mechanical_sign_reversal(c_grid, m_grid, R2)

    three = ThreeStateProtectionModel()
    t = np.linspace(0.0, 20.0, 600)
    sim = simulate_three_state_timecourse(t, c=1.0, m=0.8)
    fig, _ = plot_three_state_timecourse(sim["t"], sim["p0"], sim["p1"], sim["p2"],
                                          savepath=str(OUTDIR / "three_state_transient.png"))
    plt.close(fig)

    c_small = np.array([0.5, 1.0, 1.5])
    m_small = np.array([0.3, 0.8, 1.3])
    traces = {}
    for c in c_small:
        for m in m_small:
            traces[(float(c), float(m))] = three.responsive_fraction_timecourse(t, float(c), float(m))
    save_timecourse_csv(OUTDIR / "synthetic_timecourse.csv", t, traces)
    peaks = peak_metrics_by_condition(t, traces)

    c_grid3 = np.linspace(0.0, 2.0, 180)
    m_grid3 = np.linspace(0.0, 2.0, 180)
    C3, M3 = np.meshgrid(c_grid3, m_grid3)
    R3 = three.responsive_fraction_steady(C3, M3)
    _, mopt = find_mechanical_optima(c_grid3, m_grid3, R3)

    with (OUTDIR / "summary.txt").open("w", encoding="utf-8") as f:
        f.write("mechanopharm-minimal demo summary\n")
        f.write("================================\n\n")
        f.write("Two-state model\n")
        f.write(f"Predicted reversal concentration: {two.reversal_concentration():.4f}\n")
        f.write(f"Estimated mean EC50 at lowest m: {ec50_vals[0]:.4f}\n")
        f.write(f"Estimated mean EC50 at highest m: {ec50_vals[-1]:.4f}\n")
        f.write(f"Mechanical sign reversal detected: {reversal['has_reversal']}\n")
        f.write(f"Low-c mean slope wrt m: {reversal['low_c_mean_slope']:.4f}\n")
        f.write(f"High-c mean slope wrt m: {reversal['high_c_mean_slope']:.4f}\n\n")

        f.write("Three-state model\n")
        f.write(f"Optimal load at c=0.5: {three.optimal_load(0.5):.4f}\n")
        f.write(f"Optimal load at c=1.0: {three.optimal_load(1.0):.4f}\n")
        f.write(f"Optimal load at c=1.5: {three.optimal_load(1.5):.4f}\n\n")

        f.write("Peak metrics from selected time courses\n")
        for row in peaks:
            f.write(
                f"c={row['c']:.2f}, m={row['m']:.2f}, "
                f"peak={row['peak_value']:.4f}, t_peak={row['peak_time']:.4f}\n"
            )

        f.write("\nOptimal load path extracted from steady three-state landscape\n")
        for c, mo in zip(c_grid3[::30], mopt[::30]):
            f.write(f"c={c:.3f}, m_opt={mo:.3f}\n")

    print("Done. Outputs written to ./outputs")


if __name__ == "__main__":
    main()
