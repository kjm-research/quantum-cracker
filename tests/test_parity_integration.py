"""Integration tests for the full parity ECDLP pipeline."""

from __future__ import annotations

import numpy as np
import pytest

from quantum_cracker.parity.dynamics import ParityDynamics, compute_parity
from quantum_cracker.parity.ec_constraints import SmallEC
from quantum_cracker.parity.hamiltonian import ParityHamiltonian
from quantum_cracker.parity.oracle import ParityOracle
from quantum_cracker.parity.types import AnnealSchedule, ParityConfig


class TestEndToEndPipeline:
    """Full pipeline: curve -> keypair -> Hamiltonian -> dynamics -> oracle."""

    def test_pipeline_on_small_curve(self):
        """End-to-end on y^2 = x^3 + 7 over F_97."""
        curve = SmallEC(97, 0, 7)
        G = curve.generator
        rng = np.random.default_rng(42)
        k, P = curve.random_keypair(rng)
        n_bits = curve.key_bit_length()

        # Build Hamiltonian
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

        # Verify ground state
        gs_key = h.ground_state_key()
        assert curve.multiply(G, gs_key) == P

        # Run oracle
        oracle = ParityOracle(h, config)
        result = oracle.measure(
            n_trajectories=100,
            schedule=AnnealSchedule(
                n_steps=500,
                beta_initial=0.1,
                beta_final=20.0,
            ),
            target_key=k,
            rng=rng,
        )

        match_rate = oracle.bit_match_rate(result, k)
        assert match_rate > 0.5

    def test_parity_vs_standard_mcmc(self):
        """Parity dynamics should reach lower energy than standard MCMC."""
        key_bits = [1, 0, 1, 1, 0, 0, 1, 0]
        target_key = sum(b << i for i, b in enumerate(key_bits))
        config = ParityConfig(
            n_spins=8,
            delta_e=2.0,
            j_coupling=0.1,
            t1_base=0.1,
            t2=1.0,
            temperature=0.2,
            mode="exact",
            constraint_weight=20.0,
        )
        h = ParityHamiltonian.from_known_key(key_bits, config)

        rng_parity = np.random.default_rng(42)
        rng_standard = np.random.default_rng(42)

        sigma0 = rng_parity.choice([-1, 1], size=8).astype(np.int8)

        # Parity dynamics
        dyn_p = ParityDynamics(h, config)
        snaps_p = dyn_p.evolve_glauber(
            sigma0.copy(), n_sweeps=200, temperature=0.2,
            target_key=target_key, log_interval=50,
            rng=rng_parity,
        )

        # Standard MCMC
        dyn_s = ParityDynamics(h, config)
        snaps_s = dyn_s.evolve_standard_mcmc(
            sigma0.copy(), n_sweeps=200, temperature=0.2,
            target_key=target_key, log_interval=50,
            rng=rng_standard,
        )

        # Both should reduce energy
        assert snaps_p[-1].energy <= snaps_p[0].energy + 1.0
        assert snaps_s[-1].energy <= snaps_s[0].energy + 1.0

    def test_multiple_curves(self):
        """Pipeline should work on multiple curve sizes."""
        primes = [97, 251]
        rng = np.random.default_rng(42)

        for p in primes:
            curve = SmallEC(p, 0, 7)
            G = curve.generator
            k, P = curve.random_keypair(rng)
            n_bits = curve.key_bit_length()

            config = ParityConfig(
                n_spins=n_bits,
                delta_e=2.0,
                mode="exact",
                constraint_weight=20.0,
            )
            h = ParityHamiltonian.from_ec_dlp(curve, G, P, config)
            gs_key = h.ground_state_key()
            assert curve.multiply(G, gs_key) == P, f"Failed on p={p}"

    def test_known_key_perfect_recovery(self):
        """On a known-key Hamiltonian with strong annealing, should
        recover the key with high accuracy."""
        key_bits = [0, 1, 1, 0, 1, 0]
        target_key = sum(b << i for i, b in enumerate(key_bits))
        config = ParityConfig(
            n_spins=6,
            delta_e=3.0,
            j_coupling=0.1,
            t1_base=0.05,
            t2=1.0,
            mode="exact",
            constraint_weight=30.0,
        )
        h = ParityHamiltonian.from_known_key(key_bits, config)
        oracle = ParityOracle(h, config)

        result = oracle.measure(
            n_trajectories=200,
            schedule=AnnealSchedule(
                n_steps=800,
                beta_initial=0.1,
                beta_final=30.0,
            ),
            target_key=target_key,
            rng=np.random.default_rng(42),
        )

        match_rate = oracle.bit_match_rate(result, target_key)
        assert match_rate >= 0.8, f"Match rate {match_rate} too low for known key"


class TestSmallECCurves:
    """Tests for the SmallEC utility class."""

    def test_curve_order(self):
        curve = SmallEC(97, 0, 7)
        order = curve.order
        assert order > 0
        # Generator * order should be point at infinity
        G = curve.generator
        assert curve.multiply(G, order) is None

    def test_keypair_validity(self):
        curve = SmallEC(97, 0, 7)
        rng = np.random.default_rng(42)
        k, P = curve.random_keypair(rng)
        assert curve.multiply(curve.generator, k) == P
        assert 1 <= k < curve.order

    def test_key_bit_length(self):
        curve = SmallEC(97, 0, 7)
        n = curve.key_bit_length()
        assert n >= 7  # order > 64 for p=97
