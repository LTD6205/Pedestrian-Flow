from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray


Mode = Literal["hard", "remote"]


@dataclass(frozen=True)
class SimConfig:
    L: float = 17.3
    dt: float = 0.001
    tau: float = 0.61
    a: float = 0.36
    b: float = 0.56
    v0_mean: float = 1.24
    v0_std: float = 0.05
    e: float = 0.07
    f: float = 2.0
    mode: Mode = "hard"
    seed: int = 12345


def density_to_N(rho: float, L: float) -> int:
    return max(1, int(np.floor(rho * L)))


def required_length(v: NDArray[np.float64] | float, a: float, b: float):
    return a + b * v


def initialise_positions(
    N: int,
    L: float,
    a: float,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """
    Random initial positions on a ring with minimum spacing a.
    This follows the paper: velocities start from zero, and positions are random
    with minimum distance a.
    """
    if N * a >= L:
        raise ValueError(f"Too many pedestrians: N*a={N*a:.3f} >= L={L:.3f}")

    free_space = L - N * a
    extra_gaps = rng.dirichlet(np.ones(N)) * free_space
    gaps = a + extra_gaps

    x = np.zeros(N, dtype=float)
    x[1:] = np.cumsum(gaps[:-1])

    # Randomly rotate the configuration on the circular corridor.
    x = (x + rng.uniform(0.0, L)) % L
    x.sort()

    return x


def front_gaps_unwrapped(x: NDArray[np.float64], L: float) -> NDArray[np.float64]:
    """
    Distance from pedestrian i to pedestrian i+1 in front.

    The pedestrians are stored in fixed order without passing.
    x is allowed to grow beyond L internally.
    """
    gaps = np.empty_like(x)
    gaps[:-1] = x[1:] - x[:-1]
    gaps[-1] = x[0] + L - x[-1]
    return gaps


def compute_force(
    x: NDArray[np.float64],
    v: NDArray[np.float64],
    v0: NDArray[np.float64],
    cfg: SimConfig,
) -> NDArray[np.float64]:
    d = required_length(v, cfg.a, cfg.b)
    gap = front_gaps_unwrapped(x, cfg.L)
    drive = (v0 - v) / cfg.tau

    if cfg.mode == "hard":
        # Paper Eq. (5)
        return np.where(gap > d, drive, -v / cfg.dt)

    if cfg.mode == "remote":
        # Paper Eq. (6)
        gap_minus_d = gap - d

        repulsion = np.empty_like(v)
        safe = gap_minus_d > 1e-4
        repulsion[safe] = cfg.e / np.power(gap_minus_d[safe], cfg.f)
        repulsion[~safe] = 1e6

        G = drive - repulsion

        return np.where(v > 0.0, G, np.maximum(0.0, G))

    raise ValueError(f"Unknown mode: {cfg.mode}")


def enforce_hard_body_constraint(
    x_old: NDArray[np.float64],
    x_new: NDArray[np.float64],
    v_new: NDArray[np.float64],
    cfg: SimConfig,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Re-examination step for hard bodies without remote action.

    If after the Euler step the distance to the person in front is smaller than
    the required length, reset position and set velocity to zero.
    """
    N = len(x_new)

    changed = True
    while changed:
        changed = False
        gaps = front_gaps_unwrapped(x_new, cfg.L)

        for i in range(N - 1, -1, -1):
            d_i = required_length(v_new[i], cfg.a, cfg.b)

            if gaps[i] < d_i:
                if v_new[i] != 0.0 or x_new[i] != x_old[i]:
                    v_new[i] = 0.0
                    x_new[i] = x_old[i]
                    changed = True

    return x_new, v_new


def step(
    x: NDArray[np.float64],
    v: NDArray[np.float64],
    v0: NDArray[np.float64],
    cfg: SimConfig,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    force = compute_force(x, v, v0, cfg)

    v_new = v + cfg.dt * force
    v_new = np.clip(v_new, 0.0, v0)

    x_old = x.copy()
    x_new = x + cfg.dt * v_new

    # Important: only the hard-body model uses the re-examination correction.
    # The remote-action model uses explicit Euler.
    if cfg.mode == "hard":
        x_new, v_new = enforce_hard_body_constraint(x_old, x_new, v_new, cfg)

    return x_new, v_new


def simulate(
    N: int,
    cfg: SimConfig,
    T_relax: int = 300_000,
    T_measure: int = 300_000,
    record_every: int | None = None,
    max_records: int | None = None,
) -> dict[str, object]:
    rng = np.random.default_rng(cfg.seed)

    x = initialise_positions(N, cfg.L, cfg.a, rng)
    v = np.zeros(N, dtype=float)

    v0 = rng.normal(cfg.v0_mean, cfg.v0_std, size=N)
    v0 = np.maximum(v0, 0.05)

    measured_sum = 0.0
    measured_count = 0
    history: list[NDArray[np.float64]] = []

    total_steps = T_relax + T_measure

    for t in range(total_steps):
        x, v = step(x, v, v0, cfg)

        if t >= T_relax:
            measured_sum += float(np.mean(v))
            measured_count += 1

            if record_every is not None and (t - T_relax) % record_every == 0:
                if max_records is None or len(history) < max_records:
                    history.append(np.mod(x, cfg.L).copy())

    return {
        "rho": N / cfg.L,
        "N": N,
        "mean_velocity": measured_sum / max(1, measured_count),
        "positions": np.array(history) if history else None,
    }


def density_wave_score(positions: NDArray[np.float64] | None, L: float) -> float:
    """
    Measures how much a position history contains a dense vertical band.
    Used only to choose a paper-like seed for Fig. 3.
    """
    if positions is None or len(positions) == 0:
        return -1.0

    bins = np.linspace(0.0, L, 35)
    scores = []

    for row in positions:
        hist, _ = np.histogram(row, bins=bins)
        scores.append(float(hist.max() - hist.mean()))

    return float(np.mean(scores))