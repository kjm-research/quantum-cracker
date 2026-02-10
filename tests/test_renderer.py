"""Tests for the PyOpenGL renderer.

Note: Most GL tests are skipped in headless environments (CI).
We test the math and data preparation functions that don't require a context.
"""

import numpy as np
import pytest
import pyrr

from quantum_cracker.core.key_interface import KeyInput
from quantum_cracker.core.rip_engine import RipEngine
from quantum_cracker.core.voxel_grid import SphericalVoxelGrid
from quantum_cracker.visualization.renderer import HAS_GL, QuantumRenderer


@pytest.fixture
def small_grid():
    grid = SphericalVoxelGrid(size=10)
    grid.initialize_from_key(KeyInput(42))
    return grid


@pytest.fixture
def engine():
    e = RipEngine()
    e.initialize_from_key(KeyInput(42))
    return e


class TestRendererCreation:
    @pytest.mark.skipif(not HAS_GL, reason="No OpenGL available")
    def test_creates(self, small_grid, engine):
        renderer = QuantumRenderer(small_grid, engine, width=640, height=480)
        assert renderer.width == 640
        assert renderer.height == 480
        assert renderer.camera_distance == 3.0

    @pytest.mark.skipif(not HAS_GL, reason="No OpenGL available")
    def test_default_state(self, small_grid, engine):
        renderer = QuantumRenderer(small_grid, engine)
        assert renderer.paused is False
        assert renderer.show_voxels is True
        assert renderer.show_threads is True
        assert renderer.animation_speed == 1.0


class TestCameraMatrices:
    @pytest.mark.skipif(not HAS_GL, reason="No OpenGL available")
    def test_view_matrix_shape(self, small_grid, engine):
        renderer = QuantumRenderer(small_grid, engine)
        view = renderer._build_view_matrix()
        assert view.shape == (4, 4)

    @pytest.mark.skipif(not HAS_GL, reason="No OpenGL available")
    def test_projection_matrix_shape(self, small_grid, engine):
        renderer = QuantumRenderer(small_grid, engine)
        proj = renderer._build_projection_matrix()
        assert proj.shape == (4, 4)

    def test_pyrr_look_at(self):
        """Test that pyrr generates valid view matrices (no GL needed)."""
        eye = np.array([3.0, 1.0, 3.0], dtype=np.float32)
        target = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        view = pyrr.matrix44.create_look_at(eye, target, up)
        assert view.shape == (4, 4)
        # Determinant should be nonzero (valid transform)
        assert abs(np.linalg.det(view)) > 1e-6

    def test_pyrr_perspective(self):
        """Test that pyrr generates valid projection matrices (no GL needed)."""
        proj = pyrr.matrix44.create_perspective_projection_matrix(
            45.0, 16 / 9, 0.1, 100.0
        )
        assert proj.shape == (4, 4)
        assert abs(np.linalg.det(proj)) > 1e-6


class TestShaderSources:
    def test_shader_strings_not_empty(self):
        from quantum_cracker.visualization.shaders import (
            THREAD_FRAGMENT_SHADER,
            THREAD_VERTEX_SHADER,
            VOXEL_FRAGMENT_SHADER,
            VOXEL_VERTEX_SHADER,
        )

        assert len(VOXEL_VERTEX_SHADER) > 100
        assert len(VOXEL_FRAGMENT_SHADER) > 50
        assert len(THREAD_VERTEX_SHADER) > 100
        assert len(THREAD_FRAGMENT_SHADER) > 50

    def test_shaders_contain_version(self):
        from quantum_cracker.visualization.shaders import (
            VOXEL_VERTEX_SHADER,
            THREAD_VERTEX_SHADER,
        )

        assert "#version 330" in VOXEL_VERTEX_SHADER
        assert "#version 330" in THREAD_VERTEX_SHADER

    def test_voxel_shader_has_resonance(self):
        from quantum_cracker.visualization.shaders import VOXEL_VERTEX_SHADER

        assert "78.0" in VOXEL_VERTEX_SHADER
        assert "sin" in VOXEL_VERTEX_SHADER
        assert "cos" in VOXEL_VERTEX_SHADER
