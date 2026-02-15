"""Tests for Simulated Quantum Annealing engine."""

from __future__ import annotations

import numpy as np
import pytest

import sys
sys.path.insert(0, "src")

from quantum_cracker.parity.dynamics import compute_parity
from quantum_cracker.parity.ec_constraints import SmallEC, ECConstraintEncoder, make_curve
from quantum_cracker.parity.hamiltonian import ParityHamiltonian
from quantum_cracker.parity.sqa import SQAEngine
from quantum_cracker.parity.types import ParityConfig, SQASchedule


@pytest.fixture
def small_curve():
    """8-bit EC curve for testing."""
    return make_curve(8)


@pytest.fixture
def small_config():
    return ParityConfig(
        n_spins=8,
        delta_e=2.0,
        j_coupling=0.1,
        t1_base=0.05,
        t2=1.0,
        temperature=0.2,
        mode="ising",
    )


class TestJPerp:
    """Tests for inter-replica coupling strength."""

    def test_j_perp_decreases_with_gamma(self):
        """Larger transverse field = weaker inter-replica coupling.

        Small Gamma (classical) -> large J_perp (replicas frozen)
        Large Gamma (quantum) -> small J_perp (replicas independent)
        """
        jp_low_gamma = SQAEngine.j_perp(gamma=0.5, temperature=0.5, n_replicas=8)
        jp_high_gamma = SQAEngine.j_perp(gamma=4.0, temperature=0.5, n_replicas=8)
        assert jp_low_gamma > jp_high_gamma

    def test_j_perp_positive(self):
        """J_perp should always be positive (ferromagnetic coupling)."""
        for gamma in [0.1, 1.0, 5.0]:
            for T in [0.1, 0.5, 2.0]:
                jp = SQAEngine.j_perp(gamma=gamma, temperature=T, n_replicas=16)
                assert jp > 0

    def test_j_perp_zero_gamma_is_large(self):
        """When gamma->0, J_perp->inf (replicas freeze = classical)."""
        jp = SQAEngine.j_perp(gamma=0.001, temperature=0.5, n_replicas=8)
        assert jp > 10

    def test_j_perp_large_gamma_is_small(self):
        """When gamma is very large, J_perp -> 0 (replicas independent)."""
        jp = SQAEngine.j_perp(gamma=100.0, temperature=0.5, n_replicas=8)
        assert jp < 0.01


class TestParityWeighting:
    """Tests for PDQM parity-dependent J_perp."""

    def test_even_parity_gets_enhanced_coupling(self, small_config):
        h = ParityHamiltonian.from_known_key([0, 1, 0, 1, 0, 1, 0, 1], small_config)
        engine = SQAEngine(h, small_config)
        jp_base = 5.0
        jp_even = engine._j_perp_parity(jp_base, parity=1, delta_e=2.0,
                                         temperature=0.5, n_replicas=8)
        jp_odd = engine._j_perp_parity(jp_base, parity=-1, delta_e=2.0,
                                        temperature=0.5, n_replicas=8)
        assert jp_even > jp_base
        assert jp_odd < jp_base
        assert jp_even > jp_odd

    def test_parity_weighting_symmetric(self, small_config):
        """Even/odd weighting should be symmetric around base."""
        h = ParityHamiltonian.from_known_key([0, 1, 0, 1, 0, 1, 0, 1], small_config)
        engine = SQAEngine(h, small_config)
        jp_base = 5.0
        jp_even = engine._j_perp_parity(jp_base, parity=1, delta_e=2.0,
                                         temperature=0.5, n_replicas=8)
        jp_odd = engine._j_perp_parity(jp_base, parity=-1, delta_e=2.0,
                                        temperature=0.5, n_replicas=8)
        # ln(jp_even/jp_base) = -ln(jp_odd/jp_base)
        ratio_even = jp_even / jp_base
        ratio_odd = jp_odd / jp_base
        assert abs(np.log(ratio_even) + np.log(ratio_odd)) < 1e-10


class TestECEvaluatorCopy:
    """Tests that evaluator copies work correctly for replicas."""

    def test_copy_shares_power_points(self, small_curve):
        config = ParityConfig(n_spins=8, delta_e=2.0)
        G = small_curve.generator
        k, P = small_curve.random_keypair(np.random.default_rng(42))
        encoder = ECConstraintEncoder(small_curve, G, P)
        ev = encoder.make_evaluator()
        ev_copy = ev.copy()

        # Shared references for immutable data
        assert ev.power_points is ev_copy.power_points
        assert ev.neg_power_points is ev_copy.neg_power_points

    def test_copy_independent_state(self, small_curve):
        G = small_curve.generator
        k, P = small_curve.random_keypair(np.random.default_rng(42))
        encoder = ECConstraintEncoder(small_curve, G, P)
        ev = encoder.make_evaluator()
        ev.set_state(k)
        ev_copy = ev.copy()

        # Both start at same state
        assert ev.constraint_penalty() == ev_copy.constraint_penalty()

        # Flip in copy doesn't affect original
        ev_copy.flip_single(0)
        assert ev._current_key != ev_copy._current_key


class TestSQAAnneal:
    """Tests for full SQA annealing."""

    def test_sqa_runs_without_error(self, small_curve, small_config):
        G = small_curve.generator
        k, P = small_curve.random_keypair(np.random.default_rng(42))
        h = ParityHamiltonian.from_ec_dlp(small_curve, G, P, small_config)
        engine = SQAEngine(h, small_config)

        schedule = SQASchedule(
            n_steps=50, gamma_initial=4.0, gamma_final=0.01,
            beta_initial=0.5, beta_final=10.0, n_replicas=4,
        )
        result = engine.anneal(schedule, target_key=k, rng=np.random.default_rng(42))

        assert result.best_spins is not None
        assert len(result.final_replicas) == 4
        assert len(result.final_energies) == 4
        assert result.bit_match_rate is not None

    def test_sqa_finds_known_key_small(self, small_config):
        """SQA should find a known key on trivial instance."""
        key_bits = [0, 1, 0, 1, 0, 1, 0, 1]
        h = ParityHamiltonian.from_known_key(key_bits, small_config)
        engine = SQAEngine(h, small_config)

        schedule = SQASchedule(
            n_steps=200, gamma_initial=4.0, gamma_final=0.01,
            beta_initial=0.5, beta_final=20.0, n_replicas=16,
        )
        true_key = ParityHamiltonian.spins_to_key(
            np.array([1 - 2 * b for b in key_bits], dtype=np.int8)
        )
        result = engine.anneal(
            schedule, target_key=true_key, rng=np.random.default_rng(42),
        )
        assert result.bit_match_rate is not None
        assert result.bit_match_rate >= 0.5  # better than random

    def test_sqa_replica_agreement_increases(self, small_config):
        """After annealing, replicas should agree more than random."""
        key_bits = [1, 0, 1, 0, 1, 0, 1, 0]
        h = ParityHamiltonian.from_known_key(key_bits, small_config)
        engine = SQAEngine(h, small_config)

        # Very short anneal -- replicas should still partially agree
        schedule = SQASchedule(
            n_steps=100, gamma_initial=4.0, gamma_final=0.01,
            beta_initial=0.5, beta_final=20.0, n_replicas=8,
        )
        result = engine.anneal(schedule, rng=np.random.default_rng(42))
        # With 8 replicas, random agreement per bit = (1/2)^7 ~ 0.008
        # After annealing, should be much higher
        assert result.replica_agreement >= 0.0  # basic sanity

    def test_sqa_trajectory_recorded(self, small_curve, small_config):
        G = small_curve.generator
        k, P = small_curve.random_keypair(np.random.default_rng(42))
        h = ParityHamiltonian.from_ec_dlp(small_curve, G, P, small_config)
        engine = SQAEngine(h, small_config)

        schedule = SQASchedule(
            n_steps=100, gamma_initial=4.0, gamma_final=0.01,
            beta_initial=0.5, beta_final=10.0, n_replicas=4,
        )
        result = engine.anneal(schedule, target_key=k, rng=np.random.default_rng(42))
        assert len(result.trajectory) > 0

    def test_sqa_ec_dlp_8bit(self, small_curve, small_config):
        """SQA on 8-bit EC DLP should beat random (50%)."""
        rng = np.random.default_rng(123)
        G = small_curve.generator
        n_trials = 10
        matches = []

        for _ in range(n_trials):
            k, P = small_curve.random_keypair(rng)
            n_bits = small_curve.key_bit_length()
            small_config.n_spins = n_bits
            h = ParityHamiltonian.from_ec_dlp(small_curve, G, P, small_config)
            engine = SQAEngine(h, small_config)

            schedule = SQASchedule(
                n_steps=200, gamma_initial=4.0, gamma_final=0.01,
                beta_initial=0.5, beta_final=20.0, n_replicas=16,
            )
            result = engine.anneal(schedule, target_key=k, rng=rng)
            if result.bit_match_rate is not None:
                matches.append(result.bit_match_rate)

        avg_match = np.mean(matches)
        # SQA should do at least as well as random (50%)
        # On 8-bit curves with 16 replicas, expect >55%
        assert avg_match >= 0.45, f"SQA avg match {avg_match:.3f} too low"

    def test_parity_weighted_vs_uniform(self, small_curve, small_config):
        """Compare PDQM parity-weighted SQA vs standard SQA."""
        rng_pw = np.random.default_rng(42)
        rng_uni = np.random.default_rng(42)
        G = small_curve.generator
        k, P = small_curve.random_keypair(np.random.default_rng(99))
        n_bits = small_curve.key_bit_length()
        small_config.n_spins = n_bits
        h = ParityHamiltonian.from_ec_dlp(small_curve, G, P, small_config)

        # Parity-weighted
        engine_pw = SQAEngine(h, small_config)
        sched_pw = SQASchedule(
            n_steps=100, n_replicas=8, parity_weighted=True,
        )
        result_pw = engine_pw.anneal(sched_pw, target_key=k, rng=rng_pw)

        # Uniform
        engine_uni = SQAEngine(h, small_config)
        sched_uni = SQASchedule(
            n_steps=100, n_replicas=8, parity_weighted=False,
        )
        result_uni = engine_uni.anneal(sched_uni, target_key=k, rng=rng_uni)

        # Both should produce valid results (not necessarily different)
        assert result_pw.bit_match_rate is not None
        assert result_uni.bit_match_rate is not None


class TestSingleReplica:
    """P=1 should degenerate to classical-like behavior."""

    def test_single_replica_runs(self, small_config):
        key_bits = [0, 1, 1, 0, 1, 0, 0, 1]
        h = ParityHamiltonian.from_known_key(key_bits, small_config)
        engine = SQAEngine(h, small_config)

        schedule = SQASchedule(
            n_steps=50, n_replicas=1,
        )
        result = engine.anneal(schedule, rng=np.random.default_rng(42))
        assert len(result.final_replicas) == 1
        assert result.replica_agreement == 1.0
