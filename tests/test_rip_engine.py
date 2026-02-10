"""Tests for the 256-thread rip engine."""

import numpy as np
import pytest

from quantum_cracker.core.key_interface import KeyInput
from quantum_cracker.core.rip_engine import RipEngine
from quantum_cracker.utils.constants import OBSERVABLE_THRESHOLD_M, PLANCK_LENGTH
from quantum_cracker.utils.types import SimulationConfig


class TestInitialization:
    def test_default_config(self):
        engine = RipEngine()
        assert engine.num_threads == 256
        assert engine.radius == PLANCK_LENGTH
        assert engine.tick == 0

    def test_custom_config(self):
        cfg = SimulationConfig(num_threads=64)
        engine = RipEngine(config=cfg)
        assert engine.num_threads == 64

    def test_initialize_from_key(self):
        engine = RipEngine()
        key = KeyInput.random()
        engine.initialize_from_key(key)
        assert engine.directions.shape == (256, 3)
        assert engine.tick == 0
        assert len(engine.history) == 0

    def test_initialize_random(self):
        engine = RipEngine()
        engine.initialize_random()
        norms = np.linalg.norm(engine.directions, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-14)

    def test_gaps_computed_on_init(self):
        engine = RipEngine()
        engine.initialize_random()
        assert engine.gaps.shape == (256,)
        assert np.all(engine.gaps > 0)


class TestPositions:
    def test_positions_at_planck_scale(self):
        engine = RipEngine()
        engine.initialize_random()
        positions = engine.positions
        assert positions.shape == (256, 3)
        norms = np.linalg.norm(positions, axis=1)
        np.testing.assert_allclose(norms, PLANCK_LENGTH, rtol=1e-10)


class TestStep:
    def test_radius_increases(self):
        engine = RipEngine()
        engine.initialize_random()
        r0 = engine.radius
        engine.step()
        assert engine.radius > r0

    def test_tick_increments(self):
        engine = RipEngine()
        engine.initialize_random()
        engine.step()
        assert engine.tick == 1
        engine.step()
        assert engine.tick == 2

    def test_history_recorded(self):
        engine = RipEngine()
        engine.initialize_random()
        engine.step()
        assert len(engine.history) == 1
        assert "tick" in engine.history[0]
        assert "radius" in engine.history[0]
        assert "num_visible" in engine.history[0]


class TestRun:
    def test_run_returns_history(self):
        engine = RipEngine()
        engine.initialize_random()
        history = engine.run(10)
        assert len(history) == 10
        assert history[0]["tick"] == 1
        assert history[-1]["tick"] == 10

    def test_radius_grows_monotonically(self):
        engine = RipEngine()
        engine.initialize_random()
        history = engine.run(50)
        radii = [h["radius"] for h in history]
        assert all(radii[i] < radii[i + 1] for i in range(len(radii) - 1))


class TestVisibility:
    def test_starts_invisible(self):
        engine = RipEngine()
        engine.initialize_random()
        assert engine.num_visible == 0
        assert not engine.all_visible

    def test_visibility_requires_large_radius(self):
        # At Planck scale, gap * radius << 400nm, so nothing visible
        engine = RipEngine()
        engine.initialize_random()
        engine.run(100)
        # After 100 steps with Planck-length expansion, still invisible
        assert engine.num_visible == 0

    def test_forced_large_radius_makes_visible(self):
        engine = RipEngine()
        engine.initialize_random()
        # Manually set radius large enough that all gaps are visible
        min_gap = np.min(engine.gaps)
        engine.radius = (OBSERVABLE_THRESHOLD_M / min_gap) * 2.0
        engine._update_visibility()
        assert engine.all_visible


class TestThreadState:
    def test_get_single_thread(self):
        engine = RipEngine()
        engine.initialize_random()
        state = engine.get_thread_state(0)
        assert state.index == 0
        assert state.direction.shape == (3,)
        assert isinstance(state.gap, float)
        assert isinstance(state.visible, bool)

    def test_get_all_threads(self):
        engine = RipEngine()
        engine.initialize_random()
        states = engine.get_all_thread_states()
        assert len(states) == 256


class TestDeterminism:
    def test_same_key_same_result(self):
        key = KeyInput(42)
        e1 = RipEngine()
        e1.initialize_from_key(key)
        e1.run(10)

        e2 = RipEngine()
        e2.initialize_from_key(key)
        e2.run(10)

        assert e1.radius == e2.radius
        np.testing.assert_array_equal(e1.gaps, e2.gaps)
