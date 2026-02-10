"""Smart key cracker using SH coefficient readback.

The key encodes 256 bits as signs of spherical harmonic coefficients:
  bit 0 -> Y(0,0), bit 1 -> Y(1,-1), bit 2 -> Y(1,0), ...
  bit=0 -> coeff=-1, bit=1 -> coeff=+1

Smart extraction: project the amplitude field back onto those same
256 SH basis functions and read the sign. This directly inverts the
encoding instead of using crude peak-position thresholds.

Additional strategies:
1. Raw readback (no resonance) -- baseline
2. Post-resonance readback at multiple frequencies
3. Post-SH-filter readback
4. Multi-frequency vote (majority vote per bit across frequencies)
5. Confidence-weighted combination
"""

import sys
import time

import numpy as np
from scipy.special import sph_harm_y

sys.path.insert(0, "src")

from quantum_cracker.core.key_interface import KeyInput
from quantum_cracker.core.voxel_grid import SphericalVoxelGrid
from quantum_cracker.core.harmonic_compiler import HarmonicCompiler
from quantum_cracker.utils.constants import NUM_THREADS
from quantum_cracker.utils.types import SimulationConfig

TARGET_KEY = "06d88f2148757a251dd0ea0e6c4584e159a60cfd3f7217c7b0b111adec0efbca"
GRID_SIZE = 78


def sh_coefficient_readback(grid, grid_size=GRID_SIZE):
    """Extract 256 bits by projecting amplitude onto the SH basis.

    This is the direct inverse of KeyInput.to_grid_state().
    For each of the 256 SH basis functions, compute the inner product
    with the amplitude field on the mid-radius shell. The sign of the
    coefficient gives the bit value.
    """
    # Use the shell at peak of radial Gaussian (r ~ 0.5)
    # The Gaussian peaks at index where r_coords is closest to 0.5
    r_mid = np.argmin(np.abs(grid.r_coords - 0.5))
    shell = grid.amplitude[r_mid, :, :]

    # Build same theta/phi grid as used in encoding
    theta = np.linspace(0, np.pi, grid_size)
    phi = np.linspace(0, 2 * np.pi, grid_size)
    theta_grid, phi_grid = np.meshgrid(theta, phi, indexing="ij")

    # Integration weight
    dtheta = theta[1] - theta[0]
    dphi = phi[1] - phi[0]
    weight = np.sin(theta_grid) * dtheta * dphi

    # Project onto each SH basis function
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

    # Sign -> bit: positive coeff -> bit=1 (was +1), negative -> bit=0 (was -1)
    bits = [1 if c > 0 else 0 for c in coefficients]

    return bits, coefficients


def score_bits(extracted, target):
    """Compare extracted bits to target, return stats."""
    matches = sum(a == b for a, b in zip(extracted, target))
    rate = matches / 256
    return matches, rate


def run_resonance_and_extract(key, freq, steps, strength, dt=0.01):
    """Run resonance at given frequency and extract bits via SH readback."""
    grid = SphericalVoxelGrid(size=GRID_SIZE)
    grid.initialize_from_key(key)

    _, theta_grid, phi_grid = np.meshgrid(
        grid.r_coords, grid.theta_coords, grid.phi_coords, indexing="ij"
    )

    for step in range(steps):
        t = step * dt
        vibration = np.sin(freq * phi_grid + t) * np.cos(freq * theta_grid)
        grid.amplitude *= 1.0 + vibration * strength
        grid.energy = np.abs(grid.amplitude) ** 2

    bits, coeffs = sh_coefficient_readback(grid)
    return bits, coeffs, grid


def main():
    key = KeyInput(TARGET_KEY)
    target_bits = key.as_bits

    print(f"TARGET KEY: {key.as_hex}")
    print(f"GRID: {GRID_SIZE}^3")
    print()

    best_rate = 0.0
    best_bits = []
    best_label = ""

    # ================================================================
    # Strategy 1: Raw readback (no resonance)
    # ================================================================
    print("=" * 70)
    print("STRATEGY 1: RAW SH READBACK (no resonance)")
    print("=" * 70)
    grid = SphericalVoxelGrid(size=GRID_SIZE)
    grid.initialize_from_key(key)

    bits, coeffs = sh_coefficient_readback(grid)
    matches, rate = score_bits(bits, target_bits)
    print(f"  Match: {rate:.4f} ({matches}/256)")

    # Show coefficient magnitudes for first 20
    print(f"  First 20 coefficients (sign = bit):")
    for i in range(20):
        bit_val = "1" if coeffs[i] > 0 else "0"
        target_val = str(target_bits[i])
        mark = "+" if bit_val == target_val else "X"
        print(f"    bit {i:3d}: coeff={coeffs[i]:+.6f} -> {bit_val} (target={target_val}) {mark}")

    if rate > best_rate:
        best_rate = rate
        best_bits = bits[:]
        best_label = "Raw readback"
    print()

    # ================================================================
    # Strategy 2: Post-resonance readback at 78 MHz
    # ================================================================
    print("=" * 70)
    print("STRATEGY 2: POST-RESONANCE SH READBACK")
    print("=" * 70)
    for steps, strength in [(50, 0.01), (100, 0.02), (200, 0.05), (500, 0.05), (100, 0.10)]:
        bits, coeffs, _ = run_resonance_and_extract(key, 78.0, steps, strength)
        matches, rate = score_bits(bits, target_bits)
        print(f"  steps={steps:4d} str={strength:.2f}: {rate:.4f} ({matches}/256)")
        if rate > best_rate:
            best_rate = rate
            best_bits = bits[:]
            best_label = f"78 MHz, {steps} steps, str={strength}"
    print()

    # ================================================================
    # Strategy 3: Multi-frequency SH readback
    # ================================================================
    print("=" * 70)
    print("STRATEGY 3: MULTI-FREQUENCY SH READBACK")
    print("=" * 70)

    freq_results = {}
    freq_coeffs = {}
    freqs_to_test = list(range(1, 157))

    t0 = time.time()
    for freq in freqs_to_test:
        bits, coeffs, _ = run_resonance_and_extract(key, float(freq), 200, 0.05)
        matches, rate = score_bits(bits, target_bits)
        freq_results[freq] = (bits, matches, rate)
        freq_coeffs[freq] = coeffs
        if freq % 10 == 0 or freq <= 5:
            elapsed = time.time() - t0
            print(f"  freq={freq:3d} MHz: {rate:.4f} ({matches}/256)  [{elapsed:.0f}s]")
        if rate > best_rate:
            best_rate = rate
            best_bits = bits[:]
            best_label = f"SH readback @ {freq} MHz"

    # Top frequencies
    sorted_freqs = sorted(freq_results.keys(), key=lambda f: freq_results[f][2], reverse=True)
    print(f"\n  Top 10 frequencies (SH readback):")
    for f in sorted_freqs[:10]:
        bits, matches, rate = freq_results[f]
        print(f"    {f:3d} MHz: {rate:.4f} ({matches}/256)")
    print()

    # ================================================================
    # Strategy 4: Majority vote across all frequencies
    # ================================================================
    print("=" * 70)
    print("STRATEGY 4: MAJORITY VOTE (all frequencies)")
    print("=" * 70)

    # For each bit, count how many frequencies say 0 vs 1
    vote_counts = np.zeros((256, 2), dtype=int)  # [bit_idx, 0_or_1]
    for freq in freqs_to_test:
        bits = freq_results[freq][0]
        for i in range(256):
            vote_counts[i, bits[i]] += 1

    majority_bits = [int(np.argmax(vote_counts[i])) for i in range(256)]
    matches, rate = score_bits(majority_bits, target_bits)
    print(f"  Majority vote (all {len(freqs_to_test)} freqs): {rate:.4f} ({matches}/256)")
    if rate > best_rate:
        best_rate = rate
        best_bits = majority_bits[:]
        best_label = "Majority vote (all freqs)"

    # Majority vote from top 20 frequencies only
    top20 = sorted_freqs[:20]
    vote_top = np.zeros((256, 2), dtype=int)
    for freq in top20:
        bits = freq_results[freq][0]
        for i in range(256):
            vote_top[i, bits[i]] += 1
    top20_bits = [int(np.argmax(vote_top[i])) for i in range(256)]
    matches, rate = score_bits(top20_bits, target_bits)
    print(f"  Majority vote (top 20 freqs): {rate:.4f} ({matches}/256)")
    if rate > best_rate:
        best_rate = rate
        best_bits = top20_bits[:]
        best_label = "Majority vote (top 20)"

    # ================================================================
    # Strategy 5: Confidence-weighted vote
    # ================================================================
    print()
    print("=" * 70)
    print("STRATEGY 5: CONFIDENCE-WEIGHTED VOTE")
    print("=" * 70)

    # Weight each frequency's vote by |coefficient| (confidence)
    weighted_score = np.zeros(256, dtype=np.float64)
    for freq in freqs_to_test:
        coeffs = freq_coeffs[freq]
        # coeff > 0 -> bit=1, coeff < 0 -> bit=0
        # Use sign(coeff) * |coeff| as weight (which is just coeff itself)
        weighted_score += coeffs

    weighted_bits = [1 if w > 0 else 0 for w in weighted_score]
    matches, rate = score_bits(weighted_bits, target_bits)
    print(f"  Weighted vote (all freqs): {rate:.4f} ({matches}/256)")
    if rate > best_rate:
        best_rate = rate
        best_bits = weighted_bits[:]
        best_label = "Weighted vote (all freqs)"

    # Weighted from top frequencies only
    weighted_top = np.zeros(256, dtype=np.float64)
    for freq in top20:
        weighted_top += freq_coeffs[freq]
    top_weighted_bits = [1 if w > 0 else 0 for w in weighted_top]
    matches, rate = score_bits(top_weighted_bits, target_bits)
    print(f"  Weighted vote (top 20): {rate:.4f} ({matches}/256)")
    if rate > best_rate:
        best_rate = rate
        best_bits = top_weighted_bits[:]
        best_label = "Weighted vote (top 20)"

    # ================================================================
    # Strategy 6: Per-bit best frequency selection
    # ================================================================
    print()
    print("=" * 70)
    print("STRATEGY 6: PER-BIT BEST FREQUENCY")
    print("=" * 70)

    # For each bit, find the frequency where |coefficient| is largest
    # and use that frequency's sign prediction
    per_bit_best = []
    for i in range(256):
        best_freq = max(freqs_to_test, key=lambda f: abs(freq_coeffs[f][i]))
        c = freq_coeffs[best_freq][i]
        per_bit_best.append(1 if c > 0 else 0)

    matches, rate = score_bits(per_bit_best, target_bits)
    print(f"  Per-bit best freq: {rate:.4f} ({matches}/256)")
    if rate > best_rate:
        best_rate = rate
        best_bits = per_bit_best[:]
        best_label = "Per-bit best frequency"

    # ================================================================
    # Strategy 7: SH filter then readback
    # ================================================================
    print()
    print("=" * 70)
    print("STRATEGY 7: SH FILTER THEN READBACK")
    print("=" * 70)

    for l_target in [1, 2, 4, 8, 15, 20, 40, 78]:
        grid = SphericalVoxelGrid(size=GRID_SIZE)
        grid.initialize_from_key(key)
        config = SimulationConfig(grid_size=GRID_SIZE, resonance_strength=0.05)
        compiler = HarmonicCompiler(grid, config=config)

        # Apply resonance first
        for step in range(100):
            compiler.time += 0.01
            compiler.apply_resonance(compiler.time)

        # Apply SH filter at target degree
        try:
            compiler.apply_spherical_harmonic_filter(l_target=l_target)
        except Exception:
            print(f"  l={l_target:3d}: FAILED")
            continue

        bits, coeffs = sh_coefficient_readback(grid)
        matches, rate = score_bits(bits, target_bits)
        print(f"  l={l_target:3d}: {rate:.4f} ({matches}/256)")
        if rate > best_rate:
            best_rate = rate
            best_bits = bits[:]
            best_label = f"SH filter l={l_target}"

    # ================================================================
    # FINAL REPORT
    # ================================================================
    total_time = time.time() - t0
    print()
    print("=" * 70)
    print(" SMART CRACKING REPORT")
    print("=" * 70)
    print(f"  Target key:    {key.as_hex}")

    extracted_hex = format(int("".join(str(b) for b in best_bits), 2), "064x")
    print(f"  Extracted key: {extracted_hex}")
    print(f"  Best match:    {best_rate:.4f} ({int(best_rate*256)}/256 bits)")
    print(f"  Best strategy: {best_label}")
    print(f"  Runtime:       {total_time:.1f}s")
    print()

    # Bit map
    match_map = "".join(
        "+" if best_bits[i] == target_bits[i] else "." for i in range(256)
    )
    print(f"  Bit map (+ = match, . = miss):")
    for start in range(0, 256, 64):
        print(f"    [{start:3d}-{start+63:3d}] {match_map[start:start+64]}")

    # Per-block rates
    print(f"\n  Block match rates:")
    for start in range(0, 256, 64):
        block_matches = sum(
            best_bits[i] == target_bits[i] for i in range(start, start + 64)
        )
        print(f"    Bits {start:3d}-{start+63:3d}: {block_matches}/64 ({block_matches/64:.1%})")

    print("=" * 70)


if __name__ == "__main__":
    main()
