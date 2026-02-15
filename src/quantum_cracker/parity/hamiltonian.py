"""Parity Hamiltonian for the EC discrete logarithm problem.

Constructs an Ising Hamiltonian H = H_constraint + H_parity where:
- H_constraint penalizes spin configurations that don't satisfy k*G = P
- H_parity implements the PDQM parity energy gap and nearest-neighbor coupling

The ground state of H encodes the private key k.
"""

from __future__ import annotations

import numpy as np
import scipy.linalg
from numpy.typing import NDArray

from quantum_cracker.parity.ec_constraints import ECConstraintEncoder, SmallEC
from quantum_cracker.parity.types import ParityConfig


class ParityHamiltonian:
    """Ising Hamiltonian whose ground state is the EC private key.

    Supports three modes:
    - 'exact': Full 2^N dense matrix (N <= 20)
    - 'ising': Sparse coupling dict for MCMC (any N)
    - 'spec': Symbolic specification for quantum hardware (N = 256)
    """

    def __init__(self, config: ParityConfig) -> None:
        self.config = config
        self.n_spins = config.n_spins
        self._matrix: NDArray[np.float64] | None = None
        self._eigenvalues: NDArray[np.float64] | None = None
        self._eigenvectors: NDArray[np.float64] | None = None
        self._constraint_diagonal: NDArray[np.float64] | None = None
        self._h_fields: NDArray[np.float64] | None = None
        self._j_couplings: dict[tuple[int, int], float] = {}
        self._target_spins: NDArray[np.int8] | None = None

    @classmethod
    def from_ec_dlp(
        cls,
        curve: SmallEC,
        generator: tuple[int, int],
        public_key: tuple[int, int],
        config: ParityConfig | None = None,
    ) -> ParityHamiltonian:
        """Build Hamiltonian where the ground state is the private key.

        The constraint term penalizes every spin configuration except
        the one corresponding to k such that k*G = P.
        """
        n_bits = curve.key_bit_length()
        if config is None:
            config = ParityConfig(n_spins=n_bits)
        else:
            config.n_spins = n_bits

        h = cls(config)
        encoder = ECConstraintEncoder(curve, generator, public_key)

        if n_bits <= 20:
            h._constraint_diagonal = encoder.spin_penalty_diagonal()
        else:
            # For larger curves, use windowed constraints + pairwise couplings
            ec_couplings = encoder.pairwise_couplings()
            h._j_couplings.update(ec_couplings)

        # Add parity terms
        h._build_parity_terms()

        return h

    @classmethod
    def from_known_key(
        cls,
        key_bits: list[int],
        config: ParityConfig | None = None,
    ) -> ParityHamiltonian:
        """Build Hamiltonian from a known key (for validation).

        The ground state is set to match the given key bits.
        Useful for verifying the dynamics can find a known answer.
        """
        n = len(key_bits)
        if config is None:
            config = ParityConfig(n_spins=n)
        else:
            config.n_spins = n

        h = cls(config)

        # Target spins: bit 0 -> spin +1, bit 1 -> spin -1
        h._target_spins = np.array([1 - 2 * b for b in key_bits], dtype=np.int8)

        # Build constraint diagonal that penalizes everything except the target
        if n <= 20:
            size = 1 << n
            h._constraint_diagonal = np.ones(size, dtype=np.float64)
            # Find index of target spin configuration
            target_idx = 0
            for j in range(n):
                if key_bits[j] == 1:  # spin = -1
                    target_idx |= 1 << j
            h._constraint_diagonal[target_idx] = 0.0

        # Local fields that bias toward the target
        h._h_fields = np.array(
            [0.1 * (1 - 2 * b) for b in key_bits], dtype=np.float64
        )

        h._build_parity_terms()
        return h

    def _build_parity_terms(self) -> None:
        """Add PDQM parity Ising terms to the Hamiltonian.

        From the parity formalism (Section 4.3):
        H_parity = -Delta/2 * sum_i(sigma_i) - J * sum_{<i,j>}(sigma_i * sigma_j)
        """
        n = self.n_spins
        delta = self.config.delta_e
        j_coupling = self.config.j_coupling

        # Local field from parity energy gap
        if self._h_fields is None:
            self._h_fields = np.zeros(n, dtype=np.float64)
        self._h_fields += -delta / 2.0

        # Nearest-neighbor Ising coupling based on topology
        if self.config.coupling_topology == "chain":
            for i in range(n - 1):
                key = (i, i + 1)
                self._j_couplings[key] = (
                    self._j_couplings.get(key, 0.0) - j_coupling
                )
        elif self.config.coupling_topology == "all_to_all":
            scaled_j = j_coupling / n  # scale by 1/N for extensivity
            for i in range(n):
                for j in range(i + 1, n):
                    key = (i, j)
                    self._j_couplings[key] = (
                        self._j_couplings.get(key, 0.0) - scaled_j
                    )

    def to_matrix(self) -> NDArray[np.float64]:
        """Return the full dense Hamiltonian matrix.

        Only feasible for N <= 20 (2^20 = ~1M x 1M).
        """
        n = self.n_spins
        if n > 20:
            raise ValueError(f"N={n} too large for dense matrix (max 20)")

        if self._matrix is not None:
            return self._matrix

        size = 1 << n
        H = np.zeros((size, size), dtype=np.float64)

        # Constraint term (diagonal)
        if self._constraint_diagonal is not None:
            np.fill_diagonal(
                H, self.config.constraint_weight * self._constraint_diagonal
            )

        # Parity/Ising terms (diagonal: sigma_z interactions)
        for idx in range(size):
            spins = self._index_to_spins(idx, n)

            # Local fields: h_i * sigma_i
            if self._h_fields is not None:
                H[idx, idx] += np.dot(self._h_fields, spins)

            # Coupling terms: J_ij * sigma_i * sigma_j
            for (i, j), j_val in self._j_couplings.items():
                H[idx, idx] += j_val * spins[i] * spins[j]

        self._matrix = H
        return H

    def ground_state(self) -> tuple[NDArray[np.float64], float]:
        """Compute exact ground state via diagonalization.

        Returns (eigenvector, eigenvalue) for the lowest energy state.
        """
        H = self.to_matrix()

        if self._eigenvalues is None:
            self._eigenvalues, self._eigenvectors = scipy.linalg.eigh(H)

        gs_energy = float(self._eigenvalues[0])
        gs_vector = self._eigenvectors[:, 0]

        return gs_vector, gs_energy

    def ground_state_spins(self) -> NDArray[np.int8]:
        """Extract the spin configuration of the ground state.

        For a diagonal Hamiltonian (no transverse field), this is the
        configuration with the lowest energy.
        """
        H = self.to_matrix()
        diag = np.diag(H)
        gs_idx = int(np.argmin(diag))
        return self._index_to_spins(gs_idx, self.n_spins)

    def ground_state_key(self) -> int:
        """Extract the private key from the ground state."""
        spins = self.ground_state_spins()
        return self.spins_to_key(spins)

    def energy(self, spins: NDArray[np.int8]) -> float:
        """Evaluate H(sigma) for a given spin configuration."""
        n = self.n_spins
        e = 0.0

        # Constraint term
        if self._constraint_diagonal is not None:
            idx = self._spins_to_index(spins, n)
            e += self.config.constraint_weight * self._constraint_diagonal[idx]

        # Local fields
        if self._h_fields is not None:
            e += float(np.dot(self._h_fields, spins))

        # Couplings
        for (i, j), j_val in self._j_couplings.items():
            e += j_val * spins[i] * spins[j]

        return e

    def energy_change_single_flip(
        self, spins: NDArray[np.int8], flip_idx: int
    ) -> float:
        """Compute energy change from flipping spin at flip_idx.

        More efficient than computing full energy twice.
        """
        n = self.n_spins
        s_i = spins[flip_idx]
        dE = 0.0

        # Local field contribution: h_i * (-s_i) - h_i * s_i = -2 * h_i * s_i
        if self._h_fields is not None:
            dE += -2.0 * self._h_fields[flip_idx] * s_i

        # Coupling contributions
        for (i, j), j_val in self._j_couplings.items():
            if i == flip_idx:
                dE += -2.0 * j_val * s_i * spins[j]
            elif j == flip_idx:
                dE += -2.0 * j_val * spins[i] * s_i

        # Constraint term change (need full recompute for diagonal)
        if self._constraint_diagonal is not None:
            idx_before = self._spins_to_index(spins, n)
            flipped = spins.copy()
            flipped[flip_idx] *= -1
            idx_after = self._spins_to_index(flipped, n)
            dE += self.config.constraint_weight * (
                self._constraint_diagonal[idx_after]
                - self._constraint_diagonal[idx_before]
            )

        return dE

    def energy_change_pair_flip(
        self, spins: NDArray[np.int8], idx_a: int, idx_b: int
    ) -> float:
        """Compute energy change from flipping both spins at idx_a and idx_b.

        Pair flips preserve parity (both spins change sign).
        """
        n = self.n_spins
        s_a = spins[idx_a]
        s_b = spins[idx_b]
        dE = 0.0

        # Local field contributions
        if self._h_fields is not None:
            dE += -2.0 * self._h_fields[idx_a] * s_a
            dE += -2.0 * self._h_fields[idx_b] * s_b

        # Coupling contributions
        for (i, j), j_val in self._j_couplings.items():
            if i == idx_a and j == idx_b:
                # Both flip: sigma_a*sigma_b doesn't change
                # (-s_a)*(-s_b) = s_a*s_b, so dE from this term = 0
                pass
            elif i == idx_a:
                dE += -2.0 * j_val * s_a * spins[j]
            elif j == idx_a:
                dE += -2.0 * j_val * spins[i] * s_a
            elif i == idx_b:
                dE += -2.0 * j_val * s_b * spins[j]
            elif j == idx_b:
                dE += -2.0 * j_val * spins[i] * s_b

        # Constraint term change
        if self._constraint_diagonal is not None:
            idx_before = self._spins_to_index(spins, n)
            flipped = spins.copy()
            flipped[idx_a] *= -1
            flipped[idx_b] *= -1
            idx_after = self._spins_to_index(flipped, n)
            dE += self.config.constraint_weight * (
                self._constraint_diagonal[idx_after]
                - self._constraint_diagonal[idx_before]
            )

        return dE

    @staticmethod
    def spins_to_key(spins: NDArray[np.int8]) -> int:
        """Convert spin configuration to integer key.

        spin +1 -> bit 0, spin -1 -> bit 1.
        """
        key = 0
        for j, s in enumerate(spins):
            if s == -1:
                key |= 1 << j
        return key

    @staticmethod
    def key_to_spins(key: int, n_bits: int) -> NDArray[np.int8]:
        """Convert integer key to spin configuration."""
        spins = np.ones(n_bits, dtype=np.int8)
        for j in range(n_bits):
            if (key >> j) & 1:
                spins[j] = -1
        return spins

    @staticmethod
    def _index_to_spins(idx: int, n: int) -> NDArray[np.int8]:
        """Convert basis state index to spin array.

        Bit j of idx being 1 means spin j is -1.
        """
        spins = np.ones(n, dtype=np.int8)
        for j in range(n):
            if (idx >> j) & 1:
                spins[j] = -1
        return spins

    @staticmethod
    def _spins_to_index(spins: NDArray[np.int8], n: int) -> int:
        """Convert spin array to basis state index."""
        idx = 0
        for j in range(n):
            if spins[j] == -1:
                idx |= 1 << j
        return idx

    @property
    def ising_couplings(self) -> dict[tuple[int, int], float]:
        """Return {(i,j): J_ij} coupling dict."""
        return dict(self._j_couplings)

    @property
    def local_fields(self) -> NDArray[np.float64] | None:
        """Return local field strengths h_i."""
        return self._h_fields.copy() if self._h_fields is not None else None
