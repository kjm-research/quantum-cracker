"""Dataclass definitions for the Parity engine."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


@dataclass
class ParityConfig:
    """Configuration for parity engine simulation."""

    n_spins: int = 256
    delta_e: float = 1.0
    j_coupling: float = 0.2
    t1_base: float = 0.1
    t2: float = 1.0
    kappa: float = 0.5
    temperature: float = 0.1
    mode: str = "ising"  # 'exact', 'ising', or 'spec'
    coupling_topology: str = "chain"  # 'chain', 'all_to_all', 'random_regular'
    constraint_weight: float = 10.0


@dataclass
class DynamicsSnapshot:
    """State at a point during dynamics evolution."""

    step: int
    spins: NDArray[np.int8]
    energy: float
    parity: int  # +1 or -1
    magnetization: float
    overlap_with_target: float | None = None


@dataclass
class AnnealSchedule:
    """Quantum annealing schedule."""

    n_steps: int = 1000
    gamma_initial: float = 5.0
    gamma_final: float = 0.001
    beta_initial: float = 0.1
    beta_final: float = 10.0
    schedule_type: str = "linear"  # 'linear', 'exponential', 'parity_adaptive'


@dataclass
class AnnealResult:
    """Result from a single annealing trajectory."""

    final_spins: NDArray[np.int8]
    final_energy: float
    parity: int
    n_parity_flips: int
    trajectory: list[DynamicsSnapshot] = field(default_factory=list)


@dataclass
class OracleResult:
    """Result from the parity oracle measurement."""

    extracted_bits: list[int]
    bit_confidences: NDArray[np.float64]
    n_trajectories: int
    mean_energy: float
    parity_distribution: dict[int, int] = field(default_factory=dict)
    best_energy: float = 0.0
    best_configuration: NDArray[np.int8] | None = None
