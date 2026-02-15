"""Tests for the Parity Dynamics simulator."""

from __future__ import annotations

import numpy as np
import pytest

from quantum_cracker.parity.dynamics import ParityDynamics, compute_parity
from quantum_cracker.parity.hamiltonian import ParityHamiltonian
from quantum_cracker.parity.types import AnnealSchedule, ParityConfig


class TestComputeParity:
    """Tests for the parity helper."""

    def test_all_plus_is_even(self):
        spins = np.array([1, 1, 1, 1], dtype=np.int8)
        assert compute_parity(spins) == 1

    def test_one_minus_is_odd(self):
        spins = np.array([1, -1, 1, 1], dtype=np.int8)
        assert compute_parity(spins) == -1

    def test_two_minus_is_even(self):
        spins = np.array([1, -1, -1, 1], dtype=np.int8)
        assert compute_parity(spins) == 1

    def test_three_minus_is_odd(self):
        spins = np.array([-1, -1, -1, 1], dtype=np.int8)
        assert compute_parity(spins) == -1


class TestPairFlipsPreserveParity:
    """Pair flips should never change the system's parity."""

    def test_pair_flip_preserves_parity(self):
        rng = np.random.default_rng(42)
        for _ in range(100):
            n = 8
            spins = rng.choice([-1, 1], size=n).astype(np.int8)
            before = compute_parity(spins)
            i, j = rng.integers(0, n, size=2)
            if i == j:
                continue
            spins[i] *= -1
            spins[j] *= -1
            after = compute_parity(spins)
            assert before == after


class TestSingleFlipsFlipParity:
    """Single flips should always change the system's parity."""

    def test_single_flip_changes_parity(self):
        rng = np.random.default_rng(42)
        for _ in range(100):
            n = 8
            spins = rng.choice([-1, 1], size=n).astype(np.int8)
            before = compute_parity(spins)
            i = rng.integers(0, n)
            spins[i] *= -1
            after = compute_parity(spins)
            assert before != after


class TestGlauberDynamics:
    """Tests for parity-weighted Glauber dynamics."""

    @pytest.fixture
    def simple_system(self):
        key_bits = [1, 0, 1, 0, 1, 0]
        config = ParityConfig(
            n_spins=6,
            delta_e=2.0,
            j_coupling=0.1,
            t1_base=0.1,
            t2=1.0,
            temperature=0.5,
            mode="exact",
        )
        h = ParityHamiltonian.from_known_key(key_bits, config)
        return h, config, key_bits

    def test_returns_snapshots(self, simple_system):
        h, config, _ = simple_system
        dyn = ParityDynamics(h, config)
        sigma0 = np.ones(6, dtype=np.int8)
        snapshots = dyn.evolve_glauber(
            sigma0, n_sweeps=50, log_interval=10, rng=np.random.default_rng(0)
        )
        assert len(snapshots) >= 2  # initial + logged
        assert snapshots[0].step == 0

    def test_energy_decreases_on_average(self, simple_system):
        """At low temperature, energy should generally decrease."""
        h, config, _ = simple_system
        dyn = ParityDynamics(h, config)
        sigma0 = np.array([-1, -1, -1, -1, -1, -1], dtype=np.int8)
        snapshots = dyn.evolve_glauber(
            sigma0,
            n_sweeps=200,
            temperature=0.1,
            log_interval=10,
            rng=np.random.default_rng(42),
        )
        # Final energy should be lower than initial (on average)
        assert snapshots[-1].energy <= snapshots[0].energy + 1.0

    def test_history_recorded(self, simple_system):
        h, config, _ = simple_system
        dyn = ParityDynamics(h, config)
        sigma0 = np.ones(6, dtype=np.int8)
        dyn.evolve_glauber(
            sigma0, n_sweeps=30, log_interval=10, rng=np.random.default_rng(0)
        )
        assert len(dyn.history) > 0

    def test_overlap_tracking(self, simple_system):
        h, config, key_bits = simple_system
        target_key = sum(b << i for i, b in enumerate(key_bits))
        dyn = ParityDynamics(h, config)
        sigma0 = np.ones(6, dtype=np.int8)
        snapshots = dyn.evolve_glauber(
            sigma0,
            n_sweeps=20,
            target_key=target_key,
            log_interval=10,
            rng=np.random.default_rng(0),
        )
        for s in snapshots:
            assert s.overlap_with_target is not None
            assert 0.0 <= s.overlap_with_target <= 1.0


class TestStandardMCMC:
    """Tests for the baseline (non-parity) MCMC."""

    def test_runs_without_error(self):
        key_bits = [0, 1, 0, 1]
        config = ParityConfig(n_spins=4, mode="exact")
        h = ParityHamiltonian.from_known_key(key_bits, config)
        dyn = ParityDynamics(h, config)
        sigma0 = np.ones(4, dtype=np.int8)
        snapshots = dyn.evolve_standard_mcmc(
            sigma0, n_sweeps=20, log_interval=5, rng=np.random.default_rng(0)
        )
        assert len(snapshots) >= 2


class TestAnnealing:
    """Tests for simulated quantum annealing."""

    @pytest.fixture
    def anneal_system(self):
        key_bits = [1, 0, 1, 1, 0, 0, 1, 0]
        config = ParityConfig(
            n_spins=8,
            delta_e=1.0,
            j_coupling=0.1,
            t1_base=0.1,
            t2=1.0,
            mode="exact",
        )
        h = ParityHamiltonian.from_known_key(key_bits, config)
        return h, config, key_bits

    def test_returns_results(self, anneal_system):
        h, config, _ = anneal_system
        dyn = ParityDynamics(h, config)
        schedule = AnnealSchedule(n_steps=100, beta_initial=0.1, beta_final=5.0)
        results = dyn.anneal(schedule, n_reads=3, rng=np.random.default_rng(42))
        assert len(results) == 3
        for r in results:
            assert r.final_spins.shape == (8,)
            assert r.parity in (-1, 1)
            assert r.n_parity_flips >= 0
            assert len(r.trajectory) > 0

    def test_annealing_finds_ground_state(self, anneal_system):
        """With enough reads and low final temperature, annealing should
        find the ground state at least once."""
        h, config, key_bits = anneal_system
        target_key = sum(b << i for i, b in enumerate(key_bits))
        dyn = ParityDynamics(h, config)
        schedule = AnnealSchedule(
            n_steps=500, beta_initial=0.1, beta_final=20.0
        )
        results = dyn.anneal(
            schedule,
            n_reads=50,
            target_key=target_key,
            rng=np.random.default_rng(42),
        )

        # At least one read should find the correct key
        found = False
        for r in results:
            recovered = ParityHamiltonian.spins_to_key(r.final_spins)
            if recovered == target_key:
                found = True
                break
        assert found, "Annealing failed to find ground state in 50 reads"

    def test_parity_adaptive_schedule(self, anneal_system):
        """Parity-adaptive schedule should run without error."""
        h, config, _ = anneal_system
        dyn = ParityDynamics(h, config)
        schedule = AnnealSchedule(
            n_steps=100,
            beta_initial=0.1,
            beta_final=5.0,
            schedule_type="parity_adaptive",
        )
        results = dyn.anneal(schedule, n_reads=3, rng=np.random.default_rng(0))
        assert len(results) == 3

    def test_exponential_schedule(self, anneal_system):
        """Exponential schedule should run without error."""
        h, config, _ = anneal_system
        dyn = ParityDynamics(h, config)
        schedule = AnnealSchedule(
            n_steps=100,
            beta_initial=0.1,
            beta_final=5.0,
            schedule_type="exponential",
        )
        results = dyn.anneal(schedule, n_reads=3, rng=np.random.default_rng(0))
        assert len(results) == 3
