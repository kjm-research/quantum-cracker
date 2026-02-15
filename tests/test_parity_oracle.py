"""Tests for the Parity Oracle measurement protocol."""

from __future__ import annotations

import numpy as np
import pytest

from quantum_cracker.parity.ec_constraints import SmallEC
from quantum_cracker.parity.hamiltonian import ParityHamiltonian
from quantum_cracker.parity.oracle import ParityOracle
from quantum_cracker.parity.types import AnnealSchedule, ParityConfig


class TestParityOracle:
    """Tests for the oracle measurement."""

    @pytest.fixture
    def oracle_system(self):
        key_bits = [1, 0, 1, 1, 0, 0, 1, 0]
        config = ParityConfig(
            n_spins=8,
            delta_e=2.0,
            j_coupling=0.1,
            t1_base=0.1,
            t2=1.0,
            mode="exact",
            constraint_weight=20.0,
        )
        h = ParityHamiltonian.from_known_key(key_bits, config)
        target_key = sum(b << i for i, b in enumerate(key_bits))
        return h, config, target_key

    def test_returns_correct_bit_count(self, oracle_system):
        h, config, _ = oracle_system
        oracle = ParityOracle(h, config)
        result = oracle.measure(
            n_trajectories=10,
            schedule=AnnealSchedule(n_steps=100),
            rng=np.random.default_rng(42),
        )
        assert len(result.extracted_bits) == 8

    def test_confidences_in_range(self, oracle_system):
        h, config, _ = oracle_system
        oracle = ParityOracle(h, config)
        result = oracle.measure(
            n_trajectories=10,
            schedule=AnnealSchedule(n_steps=100),
            rng=np.random.default_rng(42),
        )
        assert all(0.0 <= c <= 1.0 + 1e-10 for c in result.bit_confidences)

    def test_parity_distribution_sums(self, oracle_system):
        h, config, _ = oracle_system
        oracle = ParityOracle(h, config)
        n_traj = 20
        result = oracle.measure(
            n_trajectories=n_traj,
            schedule=AnnealSchedule(n_steps=100),
            rng=np.random.default_rng(42),
        )
        total = result.parity_distribution.get(1, 0) + result.parity_distribution.get(-1, 0)
        assert total == n_traj

    def test_finds_correct_key_on_easy_problem(self, oracle_system):
        """With enough trajectories and strong annealing, should find the key."""
        h, config, target_key = oracle_system
        oracle = ParityOracle(h, config)
        result = oracle.measure(
            n_trajectories=100,
            schedule=AnnealSchedule(
                n_steps=500,
                beta_initial=0.1,
                beta_final=20.0,
            ),
            target_key=target_key,
            rng=np.random.default_rng(42),
        )
        match_rate = oracle.bit_match_rate(result, target_key)
        assert match_rate > 0.75, f"Match rate {match_rate} too low"

    def test_best_configuration_has_lowest_energy(self, oracle_system):
        h, config, _ = oracle_system
        oracle = ParityOracle(h, config)
        result = oracle.measure(
            n_trajectories=20,
            schedule=AnnealSchedule(n_steps=100),
            rng=np.random.default_rng(42),
        )
        assert result.best_configuration is not None
        assert result.best_energy <= result.mean_energy + 1e-10

    def test_extract_key_roundtrip(self, oracle_system):
        h, config, target_key = oracle_system
        oracle = ParityOracle(h, config)
        result = oracle.measure(
            n_trajectories=10,
            schedule=AnnealSchedule(n_steps=100),
            rng=np.random.default_rng(42),
        )
        extracted = oracle.extract_key(result)
        assert isinstance(extracted, int)
        assert 0 <= extracted < (1 << 8)


class TestOracleOnECDLP:
    """Test oracle on actual EC DLP."""

    def test_ec_oracle_above_random(self):
        """Oracle on a small EC curve should beat random chance."""
        curve = SmallEC(97, 0, 7)
        G = curve.generator
        rng = np.random.default_rng(42)
        k, P = curve.random_keypair(rng)

        n_bits = curve.key_bit_length()
        config = ParityConfig(
            n_spins=n_bits,
            delta_e=2.0,
            j_coupling=0.1,
            t1_base=0.05,
            t2=1.0,
            mode="exact",
            constraint_weight=20.0,
        )
        h = ParityHamiltonian.from_ec_dlp(curve, G, P, config)
        oracle = ParityOracle(h, config)

        result = oracle.measure(
            n_trajectories=200,
            schedule=AnnealSchedule(
                n_steps=1000,
                beta_initial=0.1,
                beta_final=30.0,
            ),
            target_key=k,
            rng=rng,
        )
        match_rate = oracle.bit_match_rate(result, k)
        # With a proper Hamiltonian, should beat random 0.5
        assert match_rate > 0.5, f"Oracle match rate {match_rate} not above random"
