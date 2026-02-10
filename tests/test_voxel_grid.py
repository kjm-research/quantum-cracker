"""Tests for the spherical voxel grid."""

import numpy as np
import pytest

from quantum_cracker.core.key_interface import KeyInput
from quantum_cracker.core.voxel_grid import SphericalVoxelGrid


class TestGridCreation:
    def test_default_size(self):
        grid = SphericalVoxelGrid()
        assert grid.size == 78

    def test_custom_size(self):
        grid = SphericalVoxelGrid(size=10)
        assert grid.size == 10

    def test_state_arrays_shape(self):
        grid = SphericalVoxelGrid(size=10)
        assert grid.amplitude.shape == (10, 10, 10)
        assert grid.phase.shape == (10, 10, 10)
        assert grid.energy.shape == (10, 10, 10)

    def test_initial_state_is_zero(self):
        grid = SphericalVoxelGrid(size=10)
        assert np.all(grid.amplitude == 0)
        assert np.all(grid.phase == 0)
        assert np.all(grid.energy == 0)

    def test_coordinate_ranges(self):
        grid = SphericalVoxelGrid(size=10)
        assert grid.r_coords[0] > 0  # Avoids r=0
        assert grid.r_coords[-1] == pytest.approx(1.0)
        assert grid.theta_coords[0] > 0  # Avoids poles
        assert grid.theta_coords[-1] < np.pi
        assert grid.phi_coords[0] == pytest.approx(0.0)


class TestInitFromKey:
    def test_sets_amplitude(self):
        grid = SphericalVoxelGrid(size=10)
        key = KeyInput.random()
        grid.initialize_from_key(key)
        assert np.any(grid.amplitude != 0)

    def test_sets_energy(self):
        grid = SphericalVoxelGrid(size=10)
        key = KeyInput.random()
        grid.initialize_from_key(key)
        assert np.any(grid.energy > 0)
        # Energy = amplitude^2
        np.testing.assert_allclose(grid.energy, np.abs(grid.amplitude) ** 2)

    def test_phase_is_zero(self):
        grid = SphericalVoxelGrid(size=10)
        key = KeyInput.random()
        grid.initialize_from_key(key)
        assert np.all(grid.phase == 0)


class TestReset:
    def test_reset_zeros_everything(self):
        grid = SphericalVoxelGrid(size=10)
        key = KeyInput.random()
        grid.initialize_from_key(key)
        grid.reset()
        assert np.all(grid.amplitude == 0)
        assert np.all(grid.phase == 0)
        assert np.all(grid.energy == 0)


class TestCartesianCoords:
    def test_shape(self):
        grid = SphericalVoxelGrid(size=10)
        coords = grid.get_cartesian_coords()
        assert coords.shape == (1000, 3)  # 10^3

    def test_cached(self):
        grid = SphericalVoxelGrid(size=10)
        c1 = grid.get_cartesian_coords()
        c2 = grid.get_cartesian_coords()
        assert c1 is c2  # Same object (cached)

    def test_cache_cleared_on_reset(self):
        grid = SphericalVoxelGrid(size=10)
        c1 = grid.get_cartesian_coords()
        grid.reset()
        c2 = grid.get_cartesian_coords()
        assert c1 is not c2  # Recomputed


class TestGetVoxel:
    def test_returns_dict(self):
        grid = SphericalVoxelGrid(size=10)
        v = grid.get_voxel(0, 0, 0)
        assert isinstance(v, dict)
        assert "amplitude" in v
        assert "phase" in v
        assert "energy" in v
        assert "r" in v
        assert "theta" in v
        assert "phi" in v

    def test_correct_values(self):
        grid = SphericalVoxelGrid(size=10)
        key = KeyInput(42)
        grid.initialize_from_key(key)
        v = grid.get_voxel(5, 5, 5)
        assert v["amplitude"] == grid.amplitude[5, 5, 5]
        assert v["r"] == grid.r_coords[5]


class TestSphericalHarmonicDecomposition:
    def test_coeffs_shape(self):
        grid = SphericalVoxelGrid(size=10)
        key = KeyInput.random()
        grid.initialize_from_key(key)
        # Use small l_max for speed
        coeffs = grid.decompose_spherical_harmonics(l_max=5)
        assert coeffs.shape == (10, 6, 11)  # (size, l_max+1, 2*l_max+1)

    def test_zero_grid_gives_zero_coeffs(self):
        grid = SphericalVoxelGrid(size=10)
        coeffs = grid.decompose_spherical_harmonics(l_max=3)
        np.testing.assert_allclose(coeffs, 0.0, atol=1e-14)

    def test_nonzero_for_initialized_grid(self):
        grid = SphericalVoxelGrid(size=10)
        key = KeyInput.random()
        grid.initialize_from_key(key)
        coeffs = grid.decompose_spherical_harmonics(l_max=5)
        assert np.any(np.abs(coeffs) > 1e-10)


class TestReconstruct:
    def test_reconstruct_sets_amplitude(self):
        grid = SphericalVoxelGrid(size=10)
        key = KeyInput.random()
        grid.initialize_from_key(key)
        coeffs = grid.decompose_spherical_harmonics(l_max=5)
        grid.reset()
        grid.reconstruct_from_sh(coeffs, l_max=5)
        assert np.any(grid.amplitude != 0)

    def test_reconstruct_sets_energy(self):
        grid = SphericalVoxelGrid(size=10)
        key = KeyInput.random()
        grid.initialize_from_key(key)
        coeffs = grid.decompose_spherical_harmonics(l_max=5)
        grid.reset()
        grid.reconstruct_from_sh(coeffs, l_max=5)
        np.testing.assert_allclose(grid.energy, np.abs(grid.amplitude) ** 2)


class TestSnapshot:
    def test_snapshot_keys(self):
        grid = SphericalVoxelGrid(size=10)
        snap = grid.snapshot()
        assert "size" in snap
        assert "amplitude_sum" in snap
        assert "energy_sum" in snap
        assert "energy_max" in snap
        assert "nonzero_voxels" in snap

    def test_snapshot_zero_grid(self):
        grid = SphericalVoxelGrid(size=10)
        snap = grid.snapshot()
        assert snap["amplitude_sum"] == 0.0
        assert snap["nonzero_voxels"] == 0
