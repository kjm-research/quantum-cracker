"""Tests for the Parity Hamiltonian."""

from __future__ import annotations

import numpy as np
import pytest

from quantum_cracker.parity.ec_constraints import SmallEC
from quantum_cracker.parity.hamiltonian import ParityHamiltonian
from quantum_cracker.parity.types import ParityConfig


class TestParityHamiltonian:
    """Core Hamiltonian construction and ground state tests."""

    def test_from_known_key_ground_state(self):
        """Ground state of known-key Hamiltonian matches the key."""
        key_bits = [1, 0, 1, 1, 0, 0, 1, 0]
        config = ParityConfig(n_spins=8, mode="exact")
        h = ParityHamiltonian.from_known_key(key_bits, config)

        gs_key = h.ground_state_key()
        expected = sum(b << i for i, b in enumerate(key_bits))
        assert gs_key == expected

    def test_from_known_key_energy_minimum(self):
        """Correct configuration has lower energy than random ones."""
        key_bits = [0, 1, 1, 0, 1, 0]
        config = ParityConfig(n_spins=6, mode="exact")
        h = ParityHamiltonian.from_known_key(key_bits, config)

        target_spins = np.array([1 - 2 * b for b in key_bits], dtype=np.int8)
        target_energy = h.energy(target_spins)

        rng = np.random.default_rng(42)
        for _ in range(50):
            random_spins = rng.choice([-1, 1], size=6).astype(np.int8)
            random_energy = h.energy(random_spins)
            assert target_energy <= random_energy + 1e-10

    def test_spin_key_roundtrip(self):
        """Spin-to-key and key-to-spin are inverses."""
        for key in [0, 1, 42, 127, 255]:
            spins = ParityHamiltonian.key_to_spins(key, 8)
            recovered = ParityHamiltonian.spins_to_key(spins)
            assert recovered == key

    def test_matrix_is_symmetric(self):
        """Hamiltonian matrix should be symmetric (Hermitian)."""
        key_bits = [1, 0, 1, 0]
        config = ParityConfig(n_spins=4, mode="exact")
        h = ParityHamiltonian.from_known_key(key_bits, config)
        H = h.to_matrix()
        np.testing.assert_allclose(H, H.T, atol=1e-12)

    def test_energy_matches_matrix_diagonal(self):
        """energy() should match the diagonal of to_matrix()."""
        key_bits = [0, 1, 0, 1, 1]
        config = ParityConfig(n_spins=5, mode="exact")
        h = ParityHamiltonian.from_known_key(key_bits, config)
        H = h.to_matrix()

        for idx in range(1 << 5):
            spins = ParityHamiltonian._index_to_spins(idx, 5)
            e_direct = h.energy(spins)
            e_matrix = H[idx, idx]
            assert abs(e_direct - e_matrix) < 1e-10

    def test_too_large_raises(self):
        """N > 20 should raise ValueError for dense matrix."""
        config = ParityConfig(n_spins=21, mode="exact")
        h = ParityHamiltonian(config)
        with pytest.raises(ValueError, match="too large"):
            h.to_matrix()


class TestECDLPHamiltonian:
    """Tests with actual elliptic curve DLP."""

    @pytest.fixture
    def small_curve(self):
        """y^2 = x^3 + 7 over F_97."""
        return SmallEC(97, 0, 7)

    def test_ground_state_is_private_key(self, small_curve):
        """The Hamiltonian ground state must be the actual private key."""
        rng = np.random.default_rng(42)
        G = small_curve.generator

        for _ in range(5):
            k, P = small_curve.random_keypair(rng)
            n_bits = small_curve.key_bit_length()
            config = ParityConfig(n_spins=n_bits, mode="exact")

            h = ParityHamiltonian.from_ec_dlp(small_curve, G, P, config)
            gs_key = h.ground_state_key()

            # The ground state key should satisfy k*G == P
            recovered_P = small_curve.multiply(G, gs_key)
            assert recovered_P == P, (
                f"Ground state key {gs_key} gives {recovered_P}, "
                f"expected {P} (true key={k})"
            )

    def test_correct_key_has_zero_constraint_penalty(self, small_curve):
        """The true private key should have zero constraint penalty."""
        rng = np.random.default_rng(123)
        G = small_curve.generator
        k, P = small_curve.random_keypair(rng)
        n_bits = small_curve.key_bit_length()

        config = ParityConfig(n_spins=n_bits, mode="exact")
        h = ParityHamiltonian.from_ec_dlp(small_curve, G, P, config)

        correct_spins = ParityHamiltonian.key_to_spins(k, n_bits)
        assert h._constraint_diagonal is not None
        idx = ParityHamiltonian._spins_to_index(correct_spins, n_bits)
        assert h._constraint_diagonal[idx] == 0.0

    def test_wrong_keys_have_nonzero_penalty(self, small_curve):
        """Random wrong keys should be penalized."""
        rng = np.random.default_rng(456)
        G = small_curve.generator
        k, P = small_curve.random_keypair(rng)
        n_bits = small_curve.key_bit_length()

        config = ParityConfig(n_spins=n_bits, mode="exact")
        h = ParityHamiltonian.from_ec_dlp(small_curve, G, P, config)

        correct_energy = h.energy(ParityHamiltonian.key_to_spins(k, n_bits))
        for _ in range(20):
            wrong_k = int(rng.integers(1, small_curve.order))
            if wrong_k == k:
                continue
            wrong_spins = ParityHamiltonian.key_to_spins(wrong_k, n_bits)
            wrong_energy = h.energy(wrong_spins)
            assert wrong_energy > correct_energy - 1e-10


class TestEnergyChangeDeltas:
    """Tests for efficient delta-energy computations."""

    @pytest.fixture
    def setup(self):
        key_bits = [1, 0, 1, 1, 0, 1]
        config = ParityConfig(n_spins=6, mode="exact")
        h = ParityHamiltonian.from_known_key(key_bits, config)
        spins = np.array([1, -1, 1, 1, -1, -1], dtype=np.int8)
        return h, spins

    def test_single_flip_delta(self, setup):
        """energy_change_single_flip matches full energy difference."""
        h, spins = setup
        for i in range(6):
            dE = h.energy_change_single_flip(spins, i)
            flipped = spins.copy()
            flipped[i] *= -1
            expected_dE = h.energy(flipped) - h.energy(spins)
            assert abs(dE - expected_dE) < 1e-10, f"Flip {i}: {dE} vs {expected_dE}"

    def test_pair_flip_delta(self, setup):
        """energy_change_pair_flip matches full energy difference."""
        h, spins = setup
        for i in range(5):
            j = i + 1
            dE = h.energy_change_pair_flip(spins, i, j)
            flipped = spins.copy()
            flipped[i] *= -1
            flipped[j] *= -1
            expected_dE = h.energy(flipped) - h.energy(spins)
            assert abs(dE - expected_dE) < 1e-10, f"Pair ({i},{j}): {dE} vs {expected_dE}"
