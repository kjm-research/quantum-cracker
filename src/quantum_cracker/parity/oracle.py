"""Parity oracle measurement protocol.

Exploits the PDQM prediction that even-parity configurations maintain
coherence exponentially longer than odd-parity ones (tau_even/tau_odd =
exp(Delta_E / kT)). The oracle runs multiple annealing trajectories,
weights final configurations by parity, and extracts key bits via
weighted majority vote.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from quantum_cracker.parity.dynamics import ParityDynamics, compute_parity
from quantum_cracker.parity.hamiltonian import ParityHamiltonian
from quantum_cracker.parity.types import (
    AnnealSchedule,
    OracleResult,
    ParityConfig,
)


class ParityOracle:
    """Measurement protocol using parity-weighted voting."""

    def __init__(
        self,
        hamiltonian: ParityHamiltonian,
        config: ParityConfig | None = None,
    ) -> None:
        self.hamiltonian = hamiltonian
        self.config = config or hamiltonian.config
        self.dynamics = ParityDynamics(hamiltonian, self.config)

    def measure(
        self,
        n_trajectories: int = 100,
        schedule: AnnealSchedule | None = None,
        target_key: int | None = None,
        rng: np.random.Generator | None = None,
    ) -> OracleResult:
        """Run the parity-weighted measurement protocol.

        1. Run n_trajectories annealing trajectories
        2. Weight each final configuration by:
           - Energy: exp(-beta * E) (Boltzmann weight)
           - Parity: even-parity gets exp(Delta_E / kT) boost
        3. Per-bit weighted majority vote
        4. Return extracted bits + confidence scores
        """
        if rng is None:
            rng = np.random.default_rng()
        if schedule is None:
            schedule = AnnealSchedule(
                n_steps=500,
                beta_initial=0.1,
                beta_final=10.0,
            )

        results = self.dynamics.anneal(
            schedule=schedule,
            n_reads=n_trajectories,
            target_key=target_key,
            rng=rng,
        )

        n = self.hamiltonian.n_spins
        beta_final = schedule.beta_final
        delta_e = self.config.delta_e

        # Compute weights for each trajectory
        energies = np.array([r.final_energy for r in results])
        parities = np.array([r.parity for r in results])

        # Boltzmann weight (shifted by min energy for numerical stability)
        e_min = energies.min()
        boltzmann = np.exp(-beta_final * (energies - e_min))

        # Parity coherence weight: even parity gets boost
        parity_boost = np.where(
            parities == 1,
            np.exp(delta_e * beta_final),
            1.0,
        )

        weights = boltzmann * parity_boost
        total_weight = weights.sum()
        if total_weight == 0:
            weights = np.ones(len(results))
            total_weight = float(len(results))

        # Per-bit weighted vote
        spin_votes = np.zeros(n, dtype=np.float64)
        for r, w in zip(results, weights):
            spin_votes += w * r.final_spins.astype(np.float64)

        spin_votes /= total_weight

        # Extract bits: spin > 0 means bit 0, spin < 0 means bit 1
        extracted_bits = [0 if v >= 0 else 1 for v in spin_votes]

        # Confidence: how decisive is the vote for each bit
        bit_confidences = np.abs(spin_votes)

        # Parity distribution
        parity_dist = {
            1: int(np.sum(parities == 1)),
            -1: int(np.sum(parities == -1)),
        }

        # Best configuration
        best_idx = int(np.argmin(energies))
        best_spins = results[best_idx].final_spins
        best_energy = float(energies[best_idx])

        return OracleResult(
            extracted_bits=extracted_bits,
            bit_confidences=bit_confidences,
            n_trajectories=n_trajectories,
            mean_energy=float(energies.mean()),
            parity_distribution=parity_dist,
            best_energy=best_energy,
            best_configuration=best_spins.copy(),
        )

    def bit_match_rate(
        self, result: OracleResult, true_key: int
    ) -> float:
        """Compute fraction of correctly extracted bits."""
        n = self.hamiltonian.n_spins
        correct = 0
        for j in range(n):
            true_bit = (true_key >> j) & 1
            if result.extracted_bits[j] == true_bit:
                correct += 1
        return correct / n

    def extract_key(self, result: OracleResult) -> int:
        """Convert oracle result to integer key."""
        key = 0
        for j, b in enumerate(result.extracted_bits):
            if b:
                key |= 1 << j
        return key
