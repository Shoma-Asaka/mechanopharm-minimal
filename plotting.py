from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def plot_two_state_landscape(c_grid: np.ndarray, m_grid: np.ndarray, response: np.ndarray, savepath: str | None = None):
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    mesh = ax.pcolormesh(c_grid, m_grid, response, shading="auto")
    fig.colorbar(mesh, ax=ax, label="Response")
    ax.set_xlabel("Normalized concentration c")
    ax.set_ylabel("Mechanical descriptor m")
    ax.set_title("Two-state response landscape")
    if savepath:
        fig.savefig(savepath, dpi=300, bbox_inches="tight")
    return fig, ax


def plot_three_state_timecourse(time, p0, p1, p2, savepath: str | None = None):
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.plot(time, p0, label="p0")
    ax.plot(time, p1, label="p1")
    ax.plot(time, p2, label="p2")
    ax.set_xlabel("Time")
    ax.set_ylabel("Occupancy")
    ax.set_title("Three-state transient redistribution")
    ax.legend(frameon=False)
    if savepath:
        fig.savefig(savepath, dpi=300, bbox_inches="tight")
    return fig, ax
