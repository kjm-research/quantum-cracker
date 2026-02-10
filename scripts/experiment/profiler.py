"""10-Key Harmonic Profiler.

Run N keys through every extraction channel and record all metrics.
Output: ~/Desktop/harmonic_profiles.json
"""

import json
import os
import sys
import time

import numpy as np
from scipy.special import sph_harm_y

sys.path.insert(0, "src")

from quantum_cracker.core.key_interface import KeyInput
from quantum_cracker.core.voxel_grid import SphericalVoxelGrid
from quantum_cracker.core.rip_engine import RipEngine
from quantum_cracker.core.harmonic_compiler import HarmonicCompiler
from quantum_cracker.utils.constants import NUM_THREADS, GRID_SIZE
from quantum_cracker.utils.math_helpers import uniform_sphere_points
from quantum_cracker.utils.types import SimulationConfig

NUM_KEYS = 10
PROFILE_GRID_SIZE = 78


def sh_coefficient_readback(grid, grid_size=PROFILE_GRID_SIZE):
    """Project amplitude onto SH basis, return 256 coefficients."""
    r_mid = np.argmin(np.abs(grid.r_coords - 0.5))
    shell = grid.amplitude[r_mid, :, :]

    theta = np.linspace(0, np.pi, grid_size)
    phi = np.linspace(0, 2 * np.pi, grid_size)
    theta_grid, phi_grid = np.meshgrid(theta, phi, indexing="ij")

    dtheta = theta[1] - theta[0]
    dphi = phi[1] - phi[0]
    weight = np.sin(theta_grid) * dtheta * dphi

    coefficients = np.zeros(NUM_THREADS, dtype=np.float64)
    bit_idx = 0
    degree = 0
    while bit_idx < NUM_THREADS:
        for m in range(-degree, degree + 1):
            if bit_idx >= NUM_THREADS:
                break
            ylm = sph_harm_y(degree, m, theta_grid, phi_grid).real
            coefficients[bit_idx] = np.sum(shell * ylm * weight)
            bit_idx += 1
        degree += 1

    bits = [1 if c > 0 else 0 for c in coefficients]
    return bits, coefficients


def thread_zflip_readback(key):
    """Thread z-flip channel: compare directions with base points."""
    base_points = uniform_sphere_points(NUM_THREADS)
    engine = RipEngine()
    engine.initialize_from_key(key)
    actual_dirs = engine.directions.copy()

    bits = []
    confidences = []
    z_base = []
    z_actual = []
    for i in range(NUM_THREADS):
        bz = base_points[i, 2]
        az = actual_dirs[i, 2]
        z_base.append(float(bz))
        z_actual.append(float(az))

        if bz > 0:
            bit = 0 if az > 0 else 1
        elif bz < 0:
            bit = 0 if az < 0 else 1
        else:
            bit = 0
        bits.append(bit)
        confidences.append(abs(bz))

    return bits, confidences, z_base, z_actual


def thread_vector_compare(key):
    """Direct vector comparison for thread recovery."""
    base_points = uniform_sphere_points(NUM_THREADS)
    engine = RipEngine()
    engine.initialize_from_key(key)
    actual_dirs = engine.directions.copy()

    bits = []
    distances_0 = []
    distances_1 = []
    for i in range(NUM_THREADS):
        base = base_points[i]
        actual = actual_dirs[i]

        option_0 = base.copy()
        option_0 /= np.linalg.norm(option_0)

        option_1 = base.copy()
        option_1[2] *= -1
        option_1 /= np.linalg.norm(option_1)

        d0 = float(np.linalg.norm(actual - option_0))
        d1 = float(np.linalg.norm(actual - option_1))

        bits.append(0 if d0 < d1 else 1)
        distances_0.append(d0)
        distances_1.append(d1)

    return bits, distances_0, distances_1


def lsq_readback(grid, grid_size=PROFILE_GRID_SIZE):
    """Least-squares SH coefficient extraction."""
    r_mid = np.argmin(np.abs(grid.r_coords - 0.5))
    shell = grid.amplitude[r_mid, :, :].ravel()

    theta = np.linspace(0, np.pi, grid_size)
    phi = np.linspace(0, 2 * np.pi, grid_size)
    theta_grid, phi_grid = np.meshgrid(theta, phi, indexing="ij")

    A = np.zeros((grid_size * grid_size, NUM_THREADS), dtype=np.float64)
    bit_idx = 0
    degree = 0
    while bit_idx < NUM_THREADS:
        for m in range(-degree, degree + 1):
            if bit_idx >= NUM_THREADS:
                break
            ylm = sph_harm_y(degree, m, theta_grid, phi_grid).real
            A[:, bit_idx] = ylm.ravel()
            bit_idx += 1
        degree += 1

    x, residuals, rank, sv = np.linalg.lstsq(A, shell, rcond=None)
    cond_number = float(sv[0] / sv[-1]) if len(sv) > 0 and sv[-1] > 0 else float("inf")
    residual_norm = float(np.linalg.norm(A @ x - shell))

    bits = [1 if c > 0 else 0 for c in x]
    return bits, x.tolist(), cond_number, residual_norm


def eigenvalue_extraction(grid, time_val=0.0):
    """Compute Hamiltonian eigenvalues on mid-radius shell."""
    config = SimulationConfig(grid_size=grid.size, resonance_strength=0.05)
    compiler = HarmonicCompiler(grid, config=config)
    compiler.time = time_val

    r_mid = np.argmin(np.abs(grid.r_coords - 0.5))
    eigenvalues = compiler.compute_hamiltonian_eigenvalues(shell_index=r_mid)
    return eigenvalues[:20].tolist()


def peak_extraction(grid, time_val=0.0):
    """Extract 78 peaks from the energy field."""
    config = SimulationConfig(grid_size=grid.size, resonance_strength=0.05)
    compiler = HarmonicCompiler(grid, config=config)
    compiler.time = time_val

    peaks = compiler.extract_peaks(num_peaks=78)
    return [
        {"r": p.r, "theta": p.theta, "phi": p.phi, "energy": p.energy, "amplitude": p.amplitude}
        for p in peaks
    ]


def grid_energy_stats(grid):
    """Compute energy field statistics."""
    energy = grid.energy
    per_shell = []
    for ir in range(grid.size):
        shell_energy = float(np.sum(energy[ir, :, :]))
        per_shell.append(shell_energy)

    return {
        "total_energy": float(np.sum(energy)),
        "mean_energy": float(np.mean(energy)),
        "std_energy": float(np.std(energy)),
        "max_energy": float(np.max(energy)),
        "per_shell_energy": per_shell,
    }


def frequency_sweep_readback(key, target_bits, freqs=None):
    """Run SH readback at multiple frequencies, return per-bit-per-freq match matrix."""
    if freqs is None:
        freqs = list(range(1, 157, 5))  # every 5th MHz for speed

    per_freq = {}
    for freq in freqs:
        grid = SphericalVoxelGrid(size=PROFILE_GRID_SIZE)
        grid.initialize_from_key(key)

        _, theta_grid, phi_grid = np.meshgrid(
            grid.r_coords, grid.theta_coords, grid.phi_coords, indexing="ij"
        )

        for step in range(200):
            t = step * 0.01
            vibration = np.sin(freq * phi_grid + t) * np.cos(freq * theta_grid)
            grid.amplitude *= 1.0 + vibration * 0.05
            grid.energy = np.abs(grid.amplitude) ** 2

        bits, coeffs = sh_coefficient_readback(grid)
        matches = [1 if bits[i] == target_bits[i] else 0 for i in range(256)]
        match_count = sum(matches)

        per_freq[str(freq)] = {
            "match_count": match_count,
            "match_rate": match_count / 256,
            "per_bit_match": matches,
        }

    return per_freq


def score_bits(extracted, target):
    matches = sum(a == b for a, b in zip(extracted, target))
    return matches, matches / 256


def profile_single_key(key_hex, key_index):
    """Run all channels on a single key, return full profile dict."""
    print(f"\n{'='*70}")
    print(f"  KEY {key_index + 1}/{NUM_KEYS}: {key_hex}")
    print(f"{'='*70}")

    key = KeyInput(key_hex)
    target_bits = key.as_bits
    t0 = time.time()

    profile = {
        "key_hex": key_hex,
        "key_bits": target_bits,
        "key_index": key_index,
    }

    # --- Channel 1: Raw SH readback ---
    print("  [1/7] SH readback (raw)...")
    grid = SphericalVoxelGrid(size=PROFILE_GRID_SIZE)
    grid.initialize_from_key(key)
    sh_bits, sh_coeffs = sh_coefficient_readback(grid)
    sh_matches, sh_rate = score_bits(sh_bits, target_bits)
    print(f"        {sh_rate:.4f} ({sh_matches}/256)")

    profile["sh_readback"] = {
        "bits": sh_bits,
        "coefficients": sh_coeffs.tolist(),
        "magnitudes": np.abs(sh_coeffs).tolist(),
        "match_count": sh_matches,
        "match_rate": sh_rate,
    }

    # --- Channel 2: Post-resonance SH readback ---
    print("  [2/7] SH readback (post-resonance)...")
    grid2 = SphericalVoxelGrid(size=PROFILE_GRID_SIZE)
    grid2.initialize_from_key(key)
    config = SimulationConfig(grid_size=PROFILE_GRID_SIZE, resonance_strength=0.05)
    compiler = HarmonicCompiler(grid2, config=config)
    for step in range(200):
        compiler.time += 0.01
        compiler.apply_resonance(compiler.time)
    sh2_bits, sh2_coeffs = sh_coefficient_readback(grid2)
    sh2_matches, sh2_rate = score_bits(sh2_bits, target_bits)
    print(f"        {sh2_rate:.4f} ({sh2_matches}/256)")

    profile["sh_post_resonance"] = {
        "bits": sh2_bits,
        "coefficients": sh2_coeffs.tolist(),
        "match_count": sh2_matches,
        "match_rate": sh2_rate,
    }

    # --- Channel 3: Thread z-flip ---
    print("  [3/7] Thread z-flip...")
    tz_bits, tz_conf, tz_base, tz_actual = thread_zflip_readback(key)
    tz_matches, tz_rate = score_bits(tz_bits, target_bits)
    print(f"        {tz_rate:.4f} ({tz_matches}/256)")

    profile["thread_zflip"] = {
        "bits": tz_bits,
        "confidences": tz_conf,
        "z_base": tz_base,
        "z_actual": tz_actual,
        "match_count": tz_matches,
        "match_rate": tz_rate,
    }

    # --- Channel 4: Thread vector compare ---
    print("  [4/7] Thread vector compare...")
    tv_bits, tv_d0, tv_d1 = thread_vector_compare(key)
    tv_matches, tv_rate = score_bits(tv_bits, target_bits)
    print(f"        {tv_rate:.4f} ({tv_matches}/256)")

    profile["thread_vector"] = {
        "bits": tv_bits,
        "distances_unflipped": tv_d0,
        "distances_flipped": tv_d1,
        "match_count": tv_matches,
        "match_rate": tv_rate,
    }

    # --- Channel 5: LSQ inversion ---
    print("  [5/7] LSQ inversion...")
    grid_lsq = SphericalVoxelGrid(size=PROFILE_GRID_SIZE)
    grid_lsq.initialize_from_key(key)
    lsq_bits, lsq_coeffs, cond_num, resid_norm = lsq_readback(grid_lsq)
    lsq_matches, lsq_rate = score_bits(lsq_bits, target_bits)
    print(f"        {lsq_rate:.4f} ({lsq_matches}/256), cond={cond_num:.2e}")

    profile["lsq_inversion"] = {
        "bits": lsq_bits,
        "coefficients": lsq_coeffs,
        "condition_number": cond_num,
        "residual_norm": resid_norm,
        "match_count": lsq_matches,
        "match_rate": lsq_rate,
    }

    # --- Channel 6: Eigenvalues ---
    print("  [6/7] Eigenvalues...")
    grid_eig = SphericalVoxelGrid(size=PROFILE_GRID_SIZE)
    grid_eig.initialize_from_key(key)
    eigenvalues = eigenvalue_extraction(grid_eig)
    print(f"        ground_state={eigenvalues[0]:.4f}")

    profile["eigenvalues"] = {
        "top_20": eigenvalues,
        "ground_state": eigenvalues[0],
    }

    # --- Channel 7: Peaks + Energy stats ---
    print("  [7/7] Peaks + energy stats...")
    grid_pk = SphericalVoxelGrid(size=PROFILE_GRID_SIZE)
    grid_pk.initialize_from_key(key)
    peaks = peak_extraction(grid_pk)
    energy_stats = grid_energy_stats(grid_pk)
    print(f"        {len(peaks)} peaks, total_energy={energy_stats['total_energy']:.4f}")

    profile["peaks"] = peaks
    profile["energy_stats"] = energy_stats

    # --- Frequency sweep (reduced set for speed) ---
    print("  [bonus] Frequency sweep (32 freqs)...")
    freq_data = frequency_sweep_readback(key, target_bits)
    best_freq = max(freq_data.keys(), key=lambda f: freq_data[f]["match_rate"])
    print(f"        best freq: {best_freq} MHz ({freq_data[best_freq]['match_rate']:.4f})")

    profile["frequency_sweep"] = freq_data

    elapsed = time.time() - t0
    profile["runtime_seconds"] = elapsed
    print(f"  Done in {elapsed:.1f}s")

    return profile


def main():
    print("=" * 70)
    print("  10-KEY HARMONIC PROFILER")
    print("=" * 70)

    # Generate 10 random keys
    keys = [KeyInput.random() for _ in range(NUM_KEYS)]
    key_hexes = [k.as_hex for k in keys]

    print(f"\nGenerated {NUM_KEYS} random keys:")
    for i, kh in enumerate(key_hexes):
        print(f"  [{i+1}] {kh}")

    profiles = []
    t_total = time.time()

    for i, kh in enumerate(key_hexes):
        profile = profile_single_key(kh, i)
        profiles.append(profile)

    total_time = time.time() - t_total

    # Save to Desktop
    output_path = os.path.expanduser("~/Desktop/harmonic_profiles.json")
    with open(output_path, "w") as f:
        json.dump(profiles, f, indent=2)

    print(f"\n{'='*70}")
    print(f"  PROFILING COMPLETE")
    print(f"{'='*70}")
    print(f"  Keys profiled: {NUM_KEYS}")
    print(f"  Total runtime: {total_time:.1f}s")
    print(f"  Output: {output_path}")

    # Quick summary
    print(f"\n  Per-key SH readback rates:")
    for p in profiles:
        print(f"    {p['key_hex'][:16]}...: {p['sh_readback']['match_rate']:.4f} "
              f"({p['sh_readback']['match_count']}/256)")

    print(f"\n  Per-key thread z-flip rates:")
    for p in profiles:
        print(f"    {p['key_hex'][:16]}...: {p['thread_zflip']['match_rate']:.4f} "
              f"({p['thread_zflip']['match_count']}/256)")

    print("=" * 70)


if __name__ == "__main__":
    main()
