"""
mcp_server/ising_simulator.py
=============================

2D Ising model Monte Carlo simulator.

This module is the heart of the physics scenario used throughout the course.
It is imported by `mcp_server/physics_tools_server.py` (the MCP server that
exposes `run_ising_simulation` as a tool to the agent), but it is fully
usable standalone — you can import `run_ising_simulation` from a notebook
or run this file directly to reproduce the phase-transition demo:

    python -m mcp_server.ising_simulator

Model
-----
Square L x L lattice, periodic boundary conditions, nearest-neighbor
ferromagnetic coupling. Units: J = 1, k_B = 1 (so temperature T is in
units of J/k_B; the exact Onsager critical temperature is
T_c = 2 / ln(1 + sqrt(2)) ≈ 2.269).

    H = -J * sum_{<i,j>} s_i s_j,     s_i in {-1, +1}

Two algorithms are provided:

* **Metropolis** — single-spin flip with the standard acceptance rule.
  Implemented as a checkerboard (red/black) *sweep*: each "step" updates
  the whole lattice in two vectorised sub-sweeps. This keeps the Python
  overhead to O(num_steps) instead of O(L^2 * num_steps), which is what
  lets a 32x32 / 10,000-step run finish in well under 5 s.

* **Wolff** — single-cluster update. Much better behaviour near T_c
  (kills most of the critical slowing-down). Implemented with an explicit
  stack in pure Python; each "step" flips exactly one cluster.

Observables returned
--------------------
* magnetization_mean, magnetization_std   (of |m| per spin, post-thermalisation)
* energy_mean, energy_std                 (of E per spin, post-thermalisation)
* specific_heat                            C_v = (<E^2> - <E>^2) * N / T^2
* susceptibility                          chi  = (<m^2> - <|m|>^2) * N / T
* final_configuration                     list[list[int]]  (the last spin grid)
* plus: algorithm, lattice_size, temperature, num_steps, thermalization_steps,
        acceptance_rate (Metropolis only), mean_cluster_size (Wolff only),
        elapsed_seconds
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Literal

import numpy as np


# ======================================================================
# Exact reference value (Onsager 1944): T_c = 2 / ln(1 + sqrt(2))
# ======================================================================
T_C_EXACT: float = 2.0 / math.log(1.0 + math.sqrt(2.0))  # ≈ 2.2691853...


# ======================================================================
# Helpers
# ======================================================================
def _initial_lattice(L: int, rng: np.random.Generator, hot: bool = True) -> np.ndarray:
    """Return an L x L array of +/-1 spins.

    hot=True   -> random ("infinite temperature") start
    hot=False  -> all aligned ("zero temperature") start
    """
    if hot:
        return rng.choice(np.array([-1, 1], dtype=np.int8), size=(L, L))
    return np.ones((L, L), dtype=np.int8)


def _energy_per_spin(spins: np.ndarray) -> float:
    """Energy per spin for the 2D Ising model with periodic BCs, J=1.

    E/N = -(1/N) * sum_{<i,j>} s_i s_j
    Each bond is counted once (right + down neighbours only).
    """
    right = spins * np.roll(spins, -1, axis=1)
    down = spins * np.roll(spins, -1, axis=0)
    N = spins.size
    return -float(right.sum() + down.sum()) / N


def _magnetization_per_spin(spins: np.ndarray) -> float:
    """Signed magnetisation per spin, m in [-1, 1]."""
    return float(spins.mean())


# ======================================================================
# Metropolis (checkerboard) sweeps — vectorised
# ======================================================================
def _metropolis_sweep(
    spins: np.ndarray,
    T: float,
    rng: np.random.Generator,
    masks: tuple[np.ndarray, np.ndarray],
) -> int:
    """Perform one full Metropolis sweep of the lattice (one "step").

    Uses a checkerboard decomposition: the red sublattice is updated in
    parallel first, then the black sublattice. Because neighbours of a
    red site are all black (and vice-versa), each sub-update is embarrassingly
    parallel and can be fully vectorised with numpy.

    Returns the number of accepted flips during this sweep.
    """
    beta = 1.0 / T
    accepted = 0

    for mask in masks:
        # neighbour sum on the full lattice (one roll per direction)
        nb_sum = (
            np.roll(spins, 1, axis=0)
            + np.roll(spins, -1, axis=0)
            + np.roll(spins, 1, axis=1)
            + np.roll(spins, -1, axis=1)
        )
        # dE for flipping site (i,j) is 2 * s_ij * nb_sum_ij
        dE = (2 * spins * nb_sum).astype(np.float64)

        # Metropolis acceptance: accept if dE <= 0, else with prob exp(-beta*dE)
        r = rng.random(size=spins.shape)
        accept = mask & ((dE <= 0) | (r < np.exp(-beta * dE)))

        spins[accept] = -spins[accept]
        accepted += int(accept.sum())

    return accepted


def _make_checkerboard_masks(L: int) -> tuple[np.ndarray, np.ndarray]:
    ii, jj = np.indices((L, L))
    red = ((ii + jj) % 2 == 0)
    black = ~red
    return red, black


def _run_metropolis(
    L: int,
    T: float,
    num_steps: int,
    thermalization_steps: int,
    rng: np.random.Generator,
) -> dict:
    spins = _initial_lattice(L, rng, hot=True)
    masks = _make_checkerboard_masks(L)
    N = L * L

    # --- thermalisation ---
    for _ in range(thermalization_steps):
        _metropolis_sweep(spins, T, rng, masks)

    # --- measurement ---
    m_samples = np.empty(num_steps, dtype=np.float64)
    e_samples = np.empty(num_steps, dtype=np.float64)
    total_accepted = 0

    for k in range(num_steps):
        total_accepted += _metropolis_sweep(spins, T, rng, masks)
        m_samples[k] = _magnetization_per_spin(spins)
        e_samples[k] = _energy_per_spin(spins)

    acceptance_rate = total_accepted / (num_steps * N) if num_steps > 0 else 0.0

    return {
        "spins": spins,
        "m_samples": m_samples,
        "e_samples": e_samples,
        "acceptance_rate": acceptance_rate,
        "mean_cluster_size": None,
    }


# ======================================================================
# Wolff single-cluster algorithm
# ======================================================================
def _wolff_step(
    spins: np.ndarray,
    p_add: float,
    rng: np.random.Generator,
) -> int:
    """One Wolff single-cluster update. Returns the cluster size."""
    L = spins.shape[0]

    # seed spin
    i0 = int(rng.integers(L))
    j0 = int(rng.integers(L))
    seed_spin = spins[i0, j0]

    cluster_mask = np.zeros_like(spins, dtype=bool)
    cluster_mask[i0, j0] = True
    stack = [(i0, j0)]
    cluster_size = 0

    while stack:
        i, j = stack.pop()
        cluster_size += 1
        # 4 neighbours with periodic BCs
        for ni, nj in (
            ((i - 1) % L, j),
            ((i + 1) % L, j),
            (i, (j - 1) % L),
            (i, (j + 1) % L),
        ):
            if (not cluster_mask[ni, nj]) and (spins[ni, nj] == seed_spin):
                if rng.random() < p_add:
                    cluster_mask[ni, nj] = True
                    stack.append((ni, nj))

    # flip the whole cluster
    spins[cluster_mask] = -spins[cluster_mask]
    return cluster_size


def _run_wolff(
    L: int,
    T: float,
    num_steps: int,
    thermalization_steps: int,
    rng: np.random.Generator,
) -> dict:
    spins = _initial_lattice(L, rng, hot=True)
    p_add = 1.0 - math.exp(-2.0 / T)  # standard Wolff bond-add probability (J=1)

    # --- thermalisation ---
    for _ in range(thermalization_steps):
        _wolff_step(spins, p_add, rng)

    # --- measurement ---
    m_samples = np.empty(num_steps, dtype=np.float64)
    e_samples = np.empty(num_steps, dtype=np.float64)
    cluster_sizes = np.empty(num_steps, dtype=np.int64)

    for k in range(num_steps):
        cluster_sizes[k] = _wolff_step(spins, p_add, rng)
        m_samples[k] = _magnetization_per_spin(spins)
        e_samples[k] = _energy_per_spin(spins)

    return {
        "spins": spins,
        "m_samples": m_samples,
        "e_samples": e_samples,
        "acceptance_rate": None,
        "mean_cluster_size": float(cluster_sizes.mean()) if num_steps > 0 else 0.0,
    }


# ======================================================================
# Public API
# ======================================================================
@dataclass
class IsingResult:
    """Typed view of the simulation result — useful when scripting in Python."""
    lattice_size: int
    temperature: float
    num_steps: int
    thermalization_steps: int
    algorithm: str
    magnetization_mean: float
    magnetization_std: float
    energy_mean: float
    energy_std: float
    specific_heat: float
    susceptibility: float
    acceptance_rate: float | None
    mean_cluster_size: float | None
    elapsed_seconds: float
    final_configuration: list[list[int]]

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def run_ising_simulation(
    lattice_size: int,
    temperature: float,
    num_steps: int,
    algorithm: Literal["metropolis", "wolff"] = "wolff",
    thermalization_steps: int | None = None,
    seed: int | None = None,
) -> dict:
    """Run a 2D Ising Monte Carlo simulation and return observables.

    Parameters
    ----------
    lattice_size : int
        Linear size L of the L x L lattice (periodic BCs). Must be >= 2.
    temperature : float
        Temperature T in units of J/k_B. Must be > 0. The exact critical
        temperature is T_c = 2 / ln(1 + sqrt(2)) ≈ 2.269.
    num_steps : int
        Number of Monte Carlo *measurement* steps. For Metropolis, one
        step = one full lattice sweep. For Wolff, one step = one single
        cluster update.
    algorithm : {"metropolis", "wolff"}, default "wolff"
        Which update scheme to use. Wolff is strongly preferred near T_c.
    thermalization_steps : int, optional
        Number of discarded steps at the start. Defaults to num_steps // 5
        (with a floor of 100 for Metropolis and 50 for Wolff).
    seed : int, optional
        Seed for reproducibility.

    Returns
    -------
    dict
        See module docstring for the schema. Values are plain Python
        floats/ints/lists so the result is JSON-serialisable (important
        because this function is exposed as an MCP tool).
    """
    # -------- input validation (this is a user-facing tool, be strict) --------
    if lattice_size < 2:
        raise ValueError(f"lattice_size must be >= 2, got {lattice_size}")
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    if num_steps < 1:
        raise ValueError(f"num_steps must be >= 1, got {num_steps}")
    algorithm = algorithm.lower()
    if algorithm not in ("metropolis", "wolff"):
        raise ValueError(
            f"algorithm must be 'metropolis' or 'wolff', got {algorithm!r}"
        )

    if thermalization_steps is None:
        floor = 100 if algorithm == "metropolis" else 50
        thermalization_steps = max(floor, num_steps // 5)

    rng = np.random.default_rng(seed)

    # -------- run --------
    t0 = time.perf_counter()
    if algorithm == "metropolis":
        raw = _run_metropolis(
            lattice_size, temperature, num_steps, thermalization_steps, rng
        )
    else:
        raw = _run_wolff(
            lattice_size, temperature, num_steps, thermalization_steps, rng
        )
    elapsed = time.perf_counter() - t0

    # -------- observables --------
    N = lattice_size * lattice_size
    m_samples: np.ndarray = raw["m_samples"]
    e_samples: np.ndarray = raw["e_samples"]
    abs_m = np.abs(m_samples)

    magnetization_mean = float(abs_m.mean())
    magnetization_std = float(abs_m.std(ddof=0))
    energy_mean = float(e_samples.mean())
    energy_std = float(e_samples.std(ddof=0))

    # C_v  = (<E^2> - <E>^2) * N / T^2   (e_samples are per spin, so multiply by N)
    specific_heat = float(e_samples.var(ddof=0) * N / (temperature ** 2))
    # chi  = (<m^2> - <|m|>^2) * N / T
    susceptibility = float(
        (np.mean(m_samples ** 2) - abs_m.mean() ** 2) * N / temperature
    )

    result = IsingResult(
        lattice_size=lattice_size,
        temperature=temperature,
        num_steps=num_steps,
        thermalization_steps=thermalization_steps,
        algorithm=algorithm,
        magnetization_mean=magnetization_mean,
        magnetization_std=magnetization_std,
        energy_mean=energy_mean,
        energy_std=energy_std,
        specific_heat=specific_heat,
        susceptibility=susceptibility,
        acceptance_rate=raw["acceptance_rate"],
        mean_cluster_size=raw["mean_cluster_size"],
        elapsed_seconds=float(elapsed),
        final_configuration=raw["spins"].astype(int).tolist(),
    )
    return result.to_dict()


# ======================================================================
# Standalone demo / smoke test
# ======================================================================
def _demo() -> None:
    """Run the phase-transition demo across T = [1.5, T_c, 3.0]."""
    print(f"Exact T_c (Onsager) = {T_C_EXACT:.6f}\n")

    print(f"{'T':>8} {'<|m|>':>10} {'std(|m|)':>10} {'<E>':>10} "
          f"{'C_v':>10} {'chi':>10} {'t [s]':>8}")
    print("-" * 78)

    for T in (1.5, T_C_EXACT, 3.0):
        r = run_ising_simulation(
            lattice_size=32,
            temperature=T,
            num_steps=10_000,
            algorithm="metropolis",
            seed=42,
        )
        print(
            f"{T:>8.4f} "
            f"{r['magnetization_mean']:>10.4f} "
            f"{r['magnetization_std']:>10.4f} "
            f"{r['energy_mean']:>10.4f} "
            f"{r['specific_heat']:>10.4f} "
            f"{r['susceptibility']:>10.4f} "
            f"{r['elapsed_seconds']:>8.2f}"
        )


if __name__ == "__main__":
    _demo()
