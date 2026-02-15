"""Parity dynamics simulator.

Implements three evolution modes for the parity Hamiltonian:
1. Exact unitary evolution (N <= 20, Schrodinger equation)
2. Glauber MCMC with PDQM-specific rates (pair hopping >> single hopping)
3. Simulated quantum annealing with parity-adaptive schedule

The PDQM-specific innovation: pair spin flips (parity-preserving) run at
rate t2 (unsuppressed), while single spin flips (parity-flipping) run at
rate t1 = t0 * exp(-Delta_E / kT) (exponentially suppressed at low T).
"""

from __future__ import annotations

import numpy as np
import scipy.linalg
from numpy.typing import NDArray

from quantum_cracker.parity.hamiltonian import ParityHamiltonian
from quantum_cracker.parity.types import (
    AnnealResult,
    AnnealSchedule,
    DynamicsSnapshot,
    ParityConfig,
)


def compute_parity(spins: NDArray[np.int8]) -> int:
    """Compute Z2 parity of a spin configuration.

    Product of all spins: +1 if even number of -1 spins, -1 if odd.
    """
    return int(np.prod(spins))


class ParityDynamics:
    """Evolve a spin system under PDQM parity dynamics."""

    def __init__(
        self,
        hamiltonian: ParityHamiltonian,
        config: ParityConfig | None = None,
    ) -> None:
        self.hamiltonian = hamiltonian
        self.config = config or hamiltonian.config
        self.history: list[DynamicsSnapshot] = []

    def _t1_effective(self, temperature: float) -> float:
        """Compute effective single-particle hopping rate.

        t1(T) = t1_base * exp(-Delta_E / kT)
        Exponentially suppressed at low temperature.
        """
        if temperature <= 0:
            return 0.0
        return self.config.t1_base * np.exp(
            -self.config.delta_e / temperature
        )

    def _snapshot(
        self,
        step: int,
        spins: NDArray[np.int8],
        target_key: int | None = None,
    ) -> DynamicsSnapshot:
        """Create a dynamics snapshot."""
        energy = self.hamiltonian.energy(spins)
        parity = compute_parity(spins)
        magnetization = float(np.mean(spins))
        overlap = None
        if target_key is not None:
            recovered = ParityHamiltonian.spins_to_key(spins)
            n = len(spins)
            # Bit overlap: fraction of matching bits
            xor = recovered ^ target_key
            matching = n - bin(xor).count("1")
            overlap = matching / n

        return DynamicsSnapshot(
            step=step,
            spins=spins.copy(),
            energy=energy,
            parity=parity,
            magnetization=magnetization,
            overlap_with_target=overlap,
        )

    def evolve_exact(
        self,
        psi0: NDArray[np.complex128],
        t_final: float,
        dt: float,
    ) -> list[DynamicsSnapshot]:
        """Exact Schrodinger evolution via eigendecomposition.

        |psi(t)> = sum_n c_n * exp(-i*E_n*t) |n>

        Only feasible for N <= 20.
        """
        H = self.hamiltonian.to_matrix()
        eigenvalues, eigenvectors = scipy.linalg.eigh(H)

        # Project initial state onto eigenbasis
        coeffs = eigenvectors.T @ psi0

        snapshots = []
        n_steps = int(t_final / dt)
        n = self.hamiltonian.n_spins

        for step in range(n_steps + 1):
            t = step * dt
            # Evolve in eigenbasis
            phases = np.exp(-1j * eigenvalues * t)
            psi_t = eigenvectors @ (coeffs * phases)

            # Probability distribution over basis states
            probs = np.abs(psi_t) ** 2

            # Most probable state
            max_idx = int(np.argmax(probs))
            spins = ParityHamiltonian._index_to_spins(max_idx, n)
            energy = float(eigenvalues @ (np.abs(coeffs) ** 2))

            snap = DynamicsSnapshot(
                step=step,
                spins=spins.copy(),
                energy=energy,
                parity=compute_parity(spins),
                magnetization=float(np.mean(spins)),
                overlap_with_target=None,
            )
            snapshots.append(snap)

        self.history = snapshots
        return snapshots

    def evolve_glauber(
        self,
        sigma0: NDArray[np.int8],
        n_sweeps: int,
        temperature: float | None = None,
        target_key: int | None = None,
        log_interval: int = 10,
        rng: np.random.Generator | None = None,
    ) -> list[DynamicsSnapshot]:
        """Glauber dynamics with PDQM-specific flip rates.

        Each sweep consists of N single-spin update attempts and
        N pair-spin update attempts. The key PDQM innovation:

        - Single-spin flips have acceptance rate multiplied by t1/t2
          (exponentially suppressed at low T, because they flip parity)
        - Pair-spin flips have standard Metropolis acceptance
          (parity-preserving, unsuppressed)
        """
        if rng is None:
            rng = np.random.default_rng()
        if temperature is None:
            temperature = self.config.temperature

        n = self.hamiltonian.n_spins
        spins = sigma0.copy()
        t1 = self._t1_effective(temperature)
        t2 = self.config.t2
        parity_suppression = t1 / t2 if t2 > 0 else 0.0
        beta = 1.0 / temperature if temperature > 0 else float("inf")

        snapshots = []
        snapshots.append(self._snapshot(0, spins, target_key))

        for sweep in range(1, n_sweeps + 1):
            # -- Single-spin updates (parity-flipping, suppressed) --
            for _ in range(n):
                i = rng.integers(0, n)
                dE = self.hamiltonian.energy_change_single_flip(spins, i)

                # Metropolis acceptance * parity suppression factor
                if dE <= 0:
                    accept_prob = parity_suppression
                else:
                    accept_prob = parity_suppression * np.exp(-beta * dE)

                if rng.random() < accept_prob:
                    spins[i] *= -1

            # -- Pair-spin updates (parity-preserving, unsuppressed) --
            for _ in range(n):
                i = rng.integers(0, n)
                j = rng.integers(0, n)
                if i == j:
                    continue
                dE = self.hamiltonian.energy_change_pair_flip(spins, i, j)

                # Standard Metropolis (no suppression for pair flips)
                if dE <= 0:
                    accept_prob = 1.0
                else:
                    accept_prob = np.exp(-beta * dE)

                if rng.random() < accept_prob:
                    spins[i] *= -1
                    spins[j] *= -1

            if sweep % log_interval == 0 or sweep == n_sweeps:
                snapshots.append(self._snapshot(sweep, spins, target_key))

        self.history = snapshots
        return snapshots

    def evolve_standard_mcmc(
        self,
        sigma0: NDArray[np.int8],
        n_sweeps: int,
        temperature: float | None = None,
        target_key: int | None = None,
        log_interval: int = 10,
        rng: np.random.Generator | None = None,
    ) -> list[DynamicsSnapshot]:
        """Standard Metropolis MCMC (no parity weighting).

        This is the BASELINE for comparison: single-spin flips only,
        no pair updates, no parity suppression. Standard textbook MCMC.
        """
        if rng is None:
            rng = np.random.default_rng()
        if temperature is None:
            temperature = self.config.temperature

        n = self.hamiltonian.n_spins
        spins = sigma0.copy()
        beta = 1.0 / temperature if temperature > 0 else float("inf")

        snapshots = []
        snapshots.append(self._snapshot(0, spins, target_key))

        for sweep in range(1, n_sweeps + 1):
            for _ in range(n):
                i = rng.integers(0, n)
                dE = self.hamiltonian.energy_change_single_flip(spins, i)

                if dE <= 0:
                    spins[i] *= -1
                elif rng.random() < np.exp(-beta * dE):
                    spins[i] *= -1

            if sweep % log_interval == 0 or sweep == n_sweeps:
                snapshots.append(self._snapshot(sweep, spins, target_key))

        self.history = snapshots
        return snapshots

    def anneal(
        self,
        schedule: AnnealSchedule | None = None,
        n_reads: int = 10,
        target_key: int | None = None,
        rng: np.random.Generator | None = None,
    ) -> list[AnnealResult]:
        """Simulated quantum annealing with parity-adaptive schedule.

        For each read:
        1. Start from random spin configuration
        2. Decrease temperature from high to low following schedule
        3. At each step, apply parity-weighted Glauber dynamics
        4. Record final configuration

        The parity-adaptive schedule adds an even-parity coherence
        boost: when the current configuration has even parity, the
        effective temperature is lowered by a factor related to Delta_E.
        """
        if rng is None:
            rng = np.random.default_rng()
        if schedule is None:
            schedule = AnnealSchedule()

        n = self.hamiltonian.n_spins
        results = []

        for read in range(n_reads):
            spins = rng.choice([-1, 1], size=n).astype(np.int8)
            trajectory: list[DynamicsSnapshot] = []
            n_parity_flips = 0
            prev_parity = compute_parity(spins)

            for step in range(schedule.n_steps):
                frac = step / max(schedule.n_steps - 1, 1)

                # Temperature schedule
                if schedule.schedule_type == "linear":
                    beta = (
                        schedule.beta_initial
                        + frac * (schedule.beta_final - schedule.beta_initial)
                    )
                elif schedule.schedule_type == "exponential":
                    log_bi = np.log(max(schedule.beta_initial, 1e-10))
                    log_bf = np.log(max(schedule.beta_final, 1e-10))
                    beta = np.exp(log_bi + frac * (log_bf - log_bi))
                else:  # parity_adaptive
                    beta = (
                        schedule.beta_initial
                        + frac * (schedule.beta_final - schedule.beta_initial)
                    )
                    # Parity coherence boost
                    current_parity = compute_parity(spins)
                    if current_parity == 1:  # even parity
                        beta *= 1.0 + self.config.delta_e * frac

                temperature = 1.0 / beta if beta > 0 else float("inf")
                t1 = self._t1_effective(temperature)
                t2 = self.config.t2
                parity_suppression = t1 / t2 if t2 > 0 else 0.0

                # Single-spin update (suppressed)
                i = rng.integers(0, n)
                dE = self.hamiltonian.energy_change_single_flip(spins, i)
                if dE <= 0:
                    accept = parity_suppression
                else:
                    accept = parity_suppression * np.exp(-beta * dE)
                if rng.random() < accept:
                    spins[i] *= -1

                # Pair-spin update (unsuppressed)
                i = rng.integers(0, n)
                j = rng.integers(0, n)
                if i != j:
                    dE = self.hamiltonian.energy_change_pair_flip(spins, i, j)
                    if dE <= 0:
                        accept = 1.0
                    else:
                        accept = np.exp(-beta * dE)
                    if rng.random() < accept:
                        spins[i] *= -1
                        spins[j] *= -1

                # Track parity flips
                new_parity = compute_parity(spins)
                if new_parity != prev_parity:
                    n_parity_flips += 1
                    prev_parity = new_parity

                # Log trajectory sparsely
                if step % max(schedule.n_steps // 20, 1) == 0:
                    trajectory.append(
                        self._snapshot(step, spins, target_key)
                    )

            final_parity = compute_parity(spins)
            final_energy = self.hamiltonian.energy(spins)

            results.append(
                AnnealResult(
                    final_spins=spins.copy(),
                    final_energy=final_energy,
                    parity=final_parity,
                    n_parity_flips=n_parity_flips,
                    trajectory=trajectory,
                )
            )

        return results
