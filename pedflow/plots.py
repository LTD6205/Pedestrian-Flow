from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _paper_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 11,
            "legend.fontsize": 9,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
        }
    )


def _save_current_figure(png_path: str) -> str:
    folder = os.path.dirname(png_path)

    if folder:
        os.makedirs(folder, exist_ok=True)

    plt.tight_layout()
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close()

    return png_path


def _plot_empirical_reference(ax, empirical_path: str = "data/1d_single_file_1_data.txt") -> None:
    """
    Plot empirical single-file data from the Jülich experiment database.

    The file header is usually:
        v [m/s] rho [1/m]

    So each numeric data row is interpreted as:
        column 1 = velocity v
        column 2 = density rho
    """
    import re

    if not os.path.exists(empirical_path):
        print(f"Empirical data file not found: {empirical_path}")
        return

    velocities = []
    densities = []

    with open(empirical_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            # Extract numeric values only.
            nums = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", line)

            # Skip header lines like: v [m/s] rho [1/m]
            if len(nums) < 2:
                continue

            try:
                v = float(nums[0])
                rho = float(nums[1])
            except ValueError:
                continue

            velocities.append(v)
            densities.append(rho)

    velocity = np.array(velocities, dtype=float)
    rho = np.array(densities, dtype=float)

    mask = (
        np.isfinite(rho)
        & np.isfinite(velocity)
        & (rho >= 0)
        & (rho <= 3)
        & (velocity >= 0)
        & (velocity <= 1.4)
    )

    ax.scatter(
        rho[mask],
        velocity[mask],
        marker="o",
        s=12,
        facecolors="none",
        edgecolors="black",
        linewidths=0.6,
        label="empirical",
    )
# ============================================================
# Fig. 1
# ============================================================

def plot_fig1_hard_body(
    csv_path: str,
    png_path: str = "output/fig1_hard_body.png",
) -> str:
    _paper_style()

    df = pd.read_csv(csv_path)
    hard_df = df[df["mode"] == "hard"]

    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    _plot_empirical_reference(ax)
    styles = {
        0.00: {"marker": "s", "label": "b=0.0"},
        0.56: {"marker": "o", "label": "b=0.56"},
        1.06: {"marker": "^", "label": "b=1.06"},
    }

    for b_value, style in styles.items():
        sub = hard_df[np.isclose(hard_df["b"], b_value)].sort_values("rho")

        if sub.empty:
            continue

        ax.scatter(
            sub["rho"],
            sub["mean_velocity"],
            marker=style["marker"],
            s=12,
            facecolors="black",
            edgecolors="black",
            linewidths=0.5,
            label=style["label"],
        )

    ax.set_xlabel(r"$\rho$ [1/m]")
    ax.set_ylabel(r"$v$ [m/s]")
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 1.4)
    ax.legend(frameon=True, fancybox=False, edgecolor="black", loc="upper right")
    ax.grid(False)

    return _save_current_figure(png_path)


# ============================================================
# Fig. 2
# ============================================================

def plot_fig2_remote_comparison(
    csv_path: str,
    png_path: str = "output/fig2_remote_comparison.png",
) -> str:
    _paper_style()

    df = pd.read_csv(csv_path)

    fig, ax = plt.subplots(figsize=(5.2, 3.8))

    cases = [
        {
            "mode": "hard",
            "b": 0.56,
            "marker": "o",
            "label": "without remote action, b=0.56",
            "face": "black",
        },
        {
            "mode": "remote",
            "b": 0.00,
            "marker": "s",
            "label": "with remote action, b=0",
            "face": "none",
        },
        {
            "mode": "remote",
            "b": 0.56,
            "marker": "o",
            "label": "with remote action, b=0.56",
            "face": "none",
        },
    ]

    for case in cases:
        sub = df[
            (df["mode"] == case["mode"])
            & np.isclose(df["b"], case["b"])
        ].sort_values("rho")

        if sub.empty:
            continue

        ax.scatter(
            sub["rho"],
            sub["mean_velocity"],
            marker=case["marker"],
            s=12,
            facecolors=case["face"],
            edgecolors="black",
            linewidths=0.7,
            label=case["label"],
        )

    ax.set_xlabel(r"$\rho$ [1/m]")
    ax.set_ylabel(r"$v$ [m/s]")
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 1.4)
    ax.legend(frameon=True, fancybox=False, edgecolor="black", loc="upper right")
    ax.grid(False)

    return _save_current_figure(png_path)


# ============================================================
# Fig. 3 helpers
# ============================================================

def _unwrap_mod_series(y: np.ndarray, L: float) -> np.ndarray:
    """
    Convert modulo positions into a visually continuous trajectory.
    """
    y = y % L
    dy = np.diff(y)
    dy = (dy + L / 2) % L - L / 2
    return np.concatenate([[y[0]], y[0] + np.cumsum(dy)])


def _find_dense_center(positions: np.ndarray, L: float, sign: float) -> float:
    """
    Find the spatial location of the densest region after applying sign.
    This helps choose a highlighted pedestrian for the jam panel.
    """
    bins = np.linspace(0.0, L, 36)
    total_hist = np.zeros(len(bins) - 1)

    for t in range(positions.shape[0]):
        row = (sign * positions[t]) % L
        hist, _ = np.histogram(row, bins=bins)
        total_hist += hist

    dense_bin = int(np.argmax(total_hist))
    return 0.5 * (bins[dense_bin] + bins[dense_bin + 1])


def _choose_marker_and_transform(
    positions: np.ndarray,
    L: float,
    prefer_jam: bool,
) -> tuple[int, float, float]:
    """
    Choose:
    - highlighted pedestrian,
    - orientation sign,
    - plot shift.

    For rho=1.16, prefer a smooth free-flow trajectory.
    For rho=1.21, prefer a trajectory that passes through the density wave.
    """
    steps, N = positions.shape

    best_marker = 0
    best_sign = 1.0
    best_y_unwrapped = _unwrap_mod_series(positions[:, 0] % L, L)
    best_score = -1e18

    for sign in [1.0, -1.0]:
        dense_center = _find_dense_center(positions, L, sign)

        for marker_id in range(N):
            y_mod = (sign * positions[:, marker_id]) % L
            y_unwrapped = _unwrap_mod_series(y_mod, L)

            total_dx = y_unwrapped[-1] - y_unwrapped[0]
            span = y_unwrapped.max() - y_unwrapped.min()

            # We want visual motion to go left -> right as time moves down.
            if total_dx <= 0:
                continue

            # Avoid trajectories that wrap across the boundary in the shown window.
            if span > 0.90 * L:
                continue

            dy = np.diff(y_unwrapped)
            max_jump = np.max(np.abs(dy)) if len(dy) else 0.0
            smooth_score = total_dx - 2.0 * max_jump - 0.1 * span

            dist = np.abs(y_mod - dense_center)
            dist = np.minimum(dist, L - dist)
            jam_score = np.sum(dist < 1.0)

            if prefer_jam:
                # For rho=1.21: choose a pedestrian interacting with the dense band.
                score = 5.0 * jam_score + 0.8 * smooth_score
            else:
                # For rho=1.16: choose smooth free-flow trajectory.
                score = smooth_score - 0.5 * jam_score

            if score > best_score:
                best_score = score
                best_marker = marker_id
                best_sign = sign
                best_y_unwrapped = y_unwrapped

    # Shift the chosen trajectory so it starts around L=4 and stays visible.
    shift = 4.0 - best_y_unwrapped[0]

    low = best_y_unwrapped.min() + shift
    high = best_y_unwrapped.max() + shift

    if low < 0.5:
        shift += 0.5 - low

    if high > L - 0.5:
        shift -= high - (L - 0.5)

    return best_marker, best_sign, shift


def _plot_position_panel(
    ax,
    positions: np.ndarray,
    L: float,
    title: str,
    prefer_jam: bool,
) -> None:
    steps, N = positions.shape

    marker_id, sign, shift = _choose_marker_and_transform(
        positions=positions,
        L=L,
        prefer_jam=prefer_jam,
    )

    for t in range(steps):
        row = (sign * positions[t] + shift) % L

        # Open circles: all pedestrians
        ax.plot(
            row,
            np.full(N, t),
            "o",
            markersize=2.2,
            markerfacecolor="none",
            markeredgecolor="black",
            markeredgewidth=0.55,
            linestyle="None",
        )

        # Filled circles: one highlighted pedestrian
        ax.plot(
            row[marker_id],
            t,
            "o",
            markersize=2.8,
            markerfacecolor="black",
            markeredgecolor="black",
            linestyle="None",
        )

    ax.set_title(title, pad=2)
    ax.set_xlabel("L")
    ax.set_xlim(0, L)
    ax.set_xticks(np.arange(0, 18, 2))
    ax.set_yticks([])
    ax.invert_yaxis()
    ax.grid(False)


# ============================================================
# Fig. 3
# ============================================================

def plot_fig3_time_development(
    positions: np.ndarray,
    L: float,
    png_path: str = "output/fig3_time_development.png",
) -> str:
    _paper_style()

    fig, ax = plt.subplots(figsize=(4.2, 4.0))

    N = positions.shape[1]
    rho = N / L

    _plot_position_panel(
        ax=ax,
        positions=positions,
        L=L,
        title=rf"$\rho$={rho:.2f} [1/m]",
        prefer_jam=rho > 1.2,
    )

    ax.set_ylabel("t")

    return _save_current_figure(png_path)


