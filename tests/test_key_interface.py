"""Tests for the key input interface."""

import numpy as np
import pytest

from quantum_cracker.core.key_interface import KeyInput


# A known test key (all zeros except last bit)
ZERO_KEY_HEX = "0" * 64
ONE_KEY_HEX = "0" * 63 + "1"
ALL_F_HEX = "f" * 64


class TestKeyParsing:
    def test_hex_string(self):
        key = KeyInput(ZERO_KEY_HEX)
        assert key.as_int == 0

    def test_hex_string_nonzero(self):
        key = KeyInput(ONE_KEY_HEX)
        assert key.as_int == 1

    def test_hex_string_max(self):
        key = KeyInput(ALL_F_HEX)
        assert key.as_int == 2**256 - 1

    def test_hex_with_prefix(self):
        key = KeyInput("0x" + ONE_KEY_HEX)
        assert key.as_int == 1

    def test_binary_string(self):
        bits = "0" * 255 + "1"
        key = KeyInput(bits)
        assert key.as_int == 1

    def test_integer(self):
        key = KeyInput(42)
        assert key.as_int == 42

    def test_integer_zero(self):
        key = KeyInput(0)
        assert key.as_int == 0

    def test_bytes(self):
        b = b"\x00" * 31 + b"\x01"
        key = KeyInput(b)
        assert key.as_int == 1

    def test_bytes_full(self):
        b = b"\xff" * 32
        key = KeyInput(b)
        assert key.as_int == 2**256 - 1


class TestKeyValidation:
    def test_reject_short_hex(self):
        with pytest.raises(ValueError):
            KeyInput("abc")

    def test_reject_long_hex(self):
        with pytest.raises(ValueError):
            KeyInput("0" * 65)

    def test_reject_wrong_bytes_length(self):
        with pytest.raises(ValueError):
            KeyInput(b"\x00" * 16)

    def test_reject_negative_int(self):
        with pytest.raises(ValueError):
            KeyInput(-1)

    def test_reject_too_large_int(self):
        with pytest.raises(ValueError):
            KeyInput(2**257)

    def test_reject_invalid_type(self):
        with pytest.raises(TypeError):
            KeyInput([1, 2, 3])  # type: ignore[arg-type]


class TestKeyConversion:
    def test_as_hex_padded(self):
        key = KeyInput(1)
        assert len(key.as_hex) == 64
        assert key.as_hex == "0" * 63 + "1"

    def test_as_bits_length(self):
        key = KeyInput(0)
        bits = key.as_bits
        assert len(bits) == 256
        assert all(b == 0 for b in bits)

    def test_as_bits_one(self):
        key = KeyInput(1)
        bits = key.as_bits
        assert bits[-1] == 1
        assert sum(bits) == 1

    def test_as_bytes_length(self):
        key = KeyInput(0)
        assert len(key.as_bytes) == 32

    def test_round_trip_hex(self):
        key1 = KeyInput.random()
        key2 = KeyInput(key1.as_hex)
        assert key1 == key2

    def test_round_trip_bytes(self):
        key1 = KeyInput.random()
        key2 = KeyInput(key1.as_bytes)
        assert key1 == key2

    def test_round_trip_int(self):
        key1 = KeyInput.random()
        key2 = KeyInput(key1.as_int)
        assert key1 == key2


class TestThreadDirections:
    def test_shape(self):
        key = KeyInput.random()
        dirs = key.to_thread_directions()
        assert dirs.shape == (256, 3)

    def test_unit_vectors(self):
        key = KeyInput.random()
        dirs = key.to_thread_directions()
        norms = np.linalg.norm(dirs, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-14)

    def test_different_keys_different_directions(self):
        k1 = KeyInput(0)
        k2 = KeyInput(2**256 - 1)
        d1 = k1.to_thread_directions()
        d2 = k2.to_thread_directions()
        # They should differ since all bits are flipped
        assert not np.allclose(d1, d2)

    def test_deterministic(self):
        key = KeyInput(42)
        d1 = key.to_thread_directions()
        d2 = key.to_thread_directions()
        np.testing.assert_array_equal(d1, d2)


class TestGridState:
    def test_shape(self):
        key = KeyInput.random()
        grid = key.to_grid_state(78)
        assert grid.shape == (78, 78, 78)

    def test_nonzero(self):
        key = KeyInput.random()
        grid = key.to_grid_state(78)
        assert np.any(grid != 0)

    def test_bounded(self):
        key = KeyInput.random()
        grid = key.to_grid_state(78)
        assert np.abs(grid).max() <= 1.0 + 1e-10

    def test_different_keys_different_grids(self):
        g1 = KeyInput(0).to_grid_state(20)
        g2 = KeyInput(2**256 - 1).to_grid_state(20)
        assert not np.allclose(g1, g2)

    def test_small_grid(self):
        # Should work with smaller grid sizes too
        key = KeyInput.random()
        grid = key.to_grid_state(10)
        assert grid.shape == (10, 10, 10)


class TestRandom:
    def test_random_creates_valid_key(self):
        key = KeyInput.random()
        assert 0 <= key.as_int <= 2**256 - 1
        assert len(key.as_hex) == 64
        assert len(key.as_bits) == 256

    def test_random_keys_differ(self):
        k1 = KeyInput.random()
        k2 = KeyInput.random()
        # Astronomically unlikely to collide
        assert k1 != k2


class TestRepr:
    def test_repr(self):
        key = KeyInput(42)
        r = repr(key)
        assert "KeyInput" in r
        assert "0x" in r
