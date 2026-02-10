"""Dual-channel key cracker.

Attack the key from two independent encodings simultaneously:
1. SH coefficient readback from the voxel grid (204/256 bits)
2. Thread direction z-flip readback from the rip engine

The key encodes bits two ways:
  Grid: bit -> +/-1 SH coefficient
  Threads: bit -> z-flip on Fibonacci spiral point

By combining both channels, ambiguous bits in one channel
may be resolved by the other.
"""

import sys
import time

import numpy as np
from scipy.special import sph_harm_y

sys.path.insert(0, "src")

from quantum_cracker.core.key_interface import KeyInput
from quantum_cracker.core.voxel_grid import SphericalVoxelGrid
from quantum_cracker.core.rip_engine import RipEngine
from quantum_cracker.utils.constants import NUM_THREADS
from quantum_cracker.utils.math_helpers import uniform_sphere_points

TARGET_KEY = "06d88f2148757a251dd0ea0e6c4584e159a60cfd3f7217c7b0b111adec0efbca"
GRID_SIZE = 78


def channel_1_sh_readback(key):
    """SH coefficient readback from the voxel grid.

    Returns (bits, confidences) where confidence is |coefficient|.
    """
    grid = SphericalVoxelGrid(size=GRID_SIZE)
    grid.initialize_from_key(key)

    # Build basis matrix
    theta = np.linspace(0, np.pi, GRID_SIZE)
    phi = np.linspace(0, 2 * np.pi, GRID_SIZE)
    theta_grid, phi_grid = np.meshgrid(theta, phi, indexing="ij")

    A = np.zeros((GRID_SIZE * GRID_SIZE, NUM_THREADS), dtype=np.float64)
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

    # LSQ on mid-radius shell
    r_mid = np.argmin(np.abs(grid.r_coords - 0.5))
    shell = grid.amplitude[r_mid, :, :].ravel()
    x, _, _, _ = np.linalg.lstsq(A, shell, rcond=None)

    bits = [1 if c > 0 else 0 for c in x]
    confidences = np.abs(x)
    return bits, confidences


def channel_2_thread_directions(key):
    """Thread direction z-flip readback from the rip engine.

    The encoding:
    1. Generate Fibonacci spiral base points (256 unit vectors)
    2. For each bit=1, flip z-component (multiply by -1)
    3. Re-normalize

    To recover: compare engine.directions[i].z with base_points[i].z.
    If signs differ, bit=1. If same, bit=0.
    """
    # Get the base points (before any bit-based flipping)
    base_points = uniform_sphere_points(NUM_THREADS)

    # Get the engine's directions (after bit-based flipping)
    engine = RipEngine()
    engine.initialize_from_key(key)
    actual_dirs = engine.directions.copy()

    bits = []
    confidences = []
    for i in range(NUM_THREADS):
        base_z = base_points[i, 2]
        actual_z = actual_dirs[i, 2]

        # If bit=0: z unchanged. If bit=1: z flipped.
        # So if sign(actual_z) != sign(base_z), bit=1
        # But for base_z near 0, this is ambiguous
        if base_z > 0:
            bit = 0 if actual_z > 0 else 1
        elif base_z < 0:
            bit = 0 if actual_z < 0 else 1
        else:
            # z=0: flipping has no effect, ambiguous
            bit = 0

        bits.append(bit)
        confidences.append(abs(base_z))  # higher |z| = more confident

    return bits, np.array(confidences)


def channel_3_thread_direct_compare(key):
    """Direct vector comparison for thread recovery.

    Compare each direction vector with both possible states
    (flipped and unflipped) and pick the closer one.
    """
    base_points = uniform_sphere_points(NUM_THREADS)

    engine = RipEngine()
    engine.initialize_from_key(key)
    actual_dirs = engine.directions.copy()

    bits = []
    confidences = []
    for i in range(NUM_THREADS):
        base = base_points[i]
        actual = actual_dirs[i]

        # Option 0: no flip (z stays same)
        option_0 = base.copy()
        option_0 /= np.linalg.norm(option_0)

        # Option 1: flip z
        option_1 = base.copy()
        option_1[2] *= -1
        option_1 /= np.linalg.norm(option_1)

        # Compare distances
        d0 = np.linalg.norm(actual - option_0)
        d1 = np.linalg.norm(actual - option_1)

        bit = 0 if d0 < d1 else 1
        confidence = abs(d0 - d1)  # larger gap = more certain

        bits.append(bit)
        confidences.append(confidence)

    return bits, np.array(confidences)


def score_bits(extracted, target):
    matches = sum(a == b for a, b in zip(extracted, target))
    return matches, matches / 256


def main():
    key = KeyInput(TARGET_KEY)
    target_bits = key.as_bits

    print(f"TARGET KEY: {key.as_hex}")
    print()

    t0 = time.time()

    # ================================================================
    # Channel 1: SH coefficient readback
    # ================================================================
    print("=" * 70)
    print("CHANNEL 1: SH COEFFICIENT READBACK")
    print("=" * 70)
    ch1_bits, ch1_conf = channel_1_sh_readback(key)
    ch1_matches, ch1_rate = score_bits(ch1_bits, target_bits)
    print(f"  Match: {ch1_rate:.4f} ({ch1_matches}/256)")
    print(f"  Confidence: mean={np.mean(ch1_conf):.6f}, median={np.median(ch1_conf):.6f}")
    print()

    # ================================================================
    # Channel 2: Thread z-flip readback
    # ================================================================
    print("=" * 70)
    print("CHANNEL 2: THREAD Z-FLIP READBACK")
    print("=" * 70)
    ch2_bits, ch2_conf = channel_2_thread_directions(key)
    ch2_matches, ch2_rate = score_bits(ch2_bits, target_bits)
    print(f"  Match: {ch2_rate:.4f} ({ch2_matches}/256)")
    print(f"  Confidence: mean={np.mean(ch2_conf):.6f}, median={np.median(ch2_conf):.6f}")
    print()

    # ================================================================
    # Channel 3: Thread direct vector comparison
    # ================================================================
    print("=" * 70)
    print("CHANNEL 3: THREAD DIRECT VECTOR COMPARISON")
    print("=" * 70)
    ch3_bits, ch3_conf = channel_3_thread_direct_compare(key)
    ch3_matches, ch3_rate = score_bits(ch3_bits, target_bits)
    print(f"  Match: {ch3_rate:.4f} ({ch3_matches}/256)")
    print(f"  Confidence: mean={np.mean(ch3_conf):.6f}, median={np.median(ch3_conf):.6f}")
    print()

    # ================================================================
    # Combined: confidence-weighted fusion
    # ================================================================
    print("=" * 70)
    print("COMBINED: CONFIDENCE-WEIGHTED FUSION")
    print("=" * 70)

    # Normalize confidences to [0, 1] range
    ch1_norm = ch1_conf / (np.max(ch1_conf) + 1e-30)
    ch2_norm = ch2_conf / (np.max(ch2_conf) + 1e-30)
    ch3_norm = ch3_conf / (np.max(ch3_conf) + 1e-30)

    combined_bits = []
    for i in range(256):
        # Weight each channel's prediction by its confidence
        score_0 = 0.0
        score_1 = 0.0

        for ch_bits, ch_conf in [(ch1_bits, ch1_norm), (ch2_bits, ch2_norm), (ch3_bits, ch3_norm)]:
            if ch_bits[i] == 0:
                score_0 += ch_conf[i]
            else:
                score_1 += ch_conf[i]

        combined_bits.append(0 if score_0 > score_1 else 1)

    comb_matches, comb_rate = score_bits(combined_bits, target_bits)
    print(f"  Combined match: {comb_rate:.4f} ({comb_matches}/256)")

    # Channel agreement analysis
    all_agree = sum(
        ch1_bits[i] == ch2_bits[i] == ch3_bits[i] for i in range(256)
    )
    print(f"  All 3 channels agree: {all_agree}/256 bits")

    agree_correct = sum(
        ch1_bits[i] == ch2_bits[i] == ch3_bits[i] == target_bits[i]
        for i in range(256)
    )
    disagree_bits = [i for i in range(256) if not (ch1_bits[i] == ch2_bits[i] == ch3_bits[i])]
    print(f"  Agreement correct:    {agree_correct}/{all_agree}")
    print(f"  Disagreement bits:    {len(disagree_bits)}")
    print()

    # Best-channel-per-bit
    best_per_bit = []
    for i in range(256):
        candidates = [
            (ch1_bits[i], ch1_norm[i]),
            (ch2_bits[i], ch2_norm[i]),
            (ch3_bits[i], ch3_norm[i]),
        ]
        best_bit = max(candidates, key=lambda x: x[1])[0]
        best_per_bit.append(best_bit)

    bpb_matches, bpb_rate = score_bits(best_per_bit, target_bits)
    print(f"  Best-channel-per-bit: {bpb_rate:.4f} ({bpb_matches}/256)")

    # Pick the best overall
    results = [
        ("CH1: SH readback", ch1_bits, ch1_rate, ch1_matches),
        ("CH2: Thread z-flip", ch2_bits, ch2_rate, ch2_matches),
        ("CH3: Thread vector", ch3_bits, ch3_rate, ch3_matches),
        ("Combined (conf-weighted)", combined_bits, comb_rate, comb_matches),
        ("Best-channel-per-bit", best_per_bit, bpb_rate, bpb_matches),
    ]

    best = max(results, key=lambda r: r[2])
    best_label, best_bits, best_rate, best_matches = best

    # ================================================================
    # FINAL REPORT
    # ================================================================
    total_time = time.time() - t0
    print()
    print("=" * 70)
    print(" DUAL-CHANNEL CRACKING REPORT")
    print("=" * 70)
    print(f"  Target key:    {key.as_hex}")
    extracted_hex = format(int("".join(str(b) for b in best_bits), 2), "064x")
    print(f"  Extracted key: {extracted_hex}")
    print(f"  Best match:    {best_rate:.4f} ({best_matches}/256 bits)")
    print(f"  Best strategy: {best_label}")
    print()

    print(f"  Summary:")
    for label, bits, rate, matches in results:
        print(f"    {label:30s}: {rate:.4f} ({matches}/256)")

    print(f"\n  Runtime: {total_time:.1f}s")
    print()

    match_map = "".join(
        "+" if best_bits[i] == target_bits[i] else "." for i in range(256)
    )
    print(f"  Bit map (+ = match, . = miss):")
    for start in range(0, 256, 64):
        print(f"    [{start:3d}-{start+63:3d}] {match_map[start:start+64]}")

    print(f"\n  Block match rates:")
    for start in range(0, 256, 64):
        block = sum(best_bits[i] == target_bits[i] for i in range(start, start + 64))
        print(f"    Bits {start:3d}-{start+63:3d}: {block}/64 ({block/64:.1%})")

    print("=" * 70)


if __name__ == "__main__":
    main()
