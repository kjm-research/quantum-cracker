"""Frequency-to-bit mapping analysis.

For each frequency, run resonance and track which specific bit positions
are correctly resolved. Then find the optimal frequency per bit and
compute the combined best-of-all-frequencies accuracy.
"""

import csv
import os
import sys
import time

import numpy as np

sys.path.insert(0, "src")

from quantum_cracker.core.key_interface import KeyInput
from quantum_cracker.core.voxel_grid import SphericalVoxelGrid
from quantum_cracker.core.harmonic_compiler import HarmonicCompiler
from quantum_cracker.analysis.metrics import MetricExtractor
from quantum_cracker.analysis.validation import Validator
from quantum_cracker.utils.types import SimulationConfig

TARGET_KEY = "06d88f2148757a251dd0ea0e6c4584e159a60cfd3f7217c7b0b111adec0efbca"
GRID_SIZE = 78

# Frequencies to sweep
FREQ_START = 1
FREQ_END = 156
FREQ_STEP = 1

# Resonance parameters per frequency
STEPS = 300
DT = 0.01
STRENGTH = 0.10


def run_at_frequency(key, freq, target_bits):
    """Run resonance at a single frequency, return per-bit match array."""
    grid = SphericalVoxelGrid(size=GRID_SIZE)
    grid.initialize_from_key(key)

    config = SimulationConfig(grid_size=GRID_SIZE, resonance_strength=STRENGTH)
    compiler = HarmonicCompiler(grid, config=config)

    # Build custom vibration at this frequency
    _, theta_grid, phi_grid = np.meshgrid(
        grid.r_coords, grid.theta_coords, grid.phi_coords, indexing="ij"
    )

    for step in range(STEPS):
        t = step * DT
        vibration = np.sin(freq * phi_grid + t) * np.cos(freq * theta_grid)
        grid.amplitude *= 1.0 + vibration * STRENGTH
        grid.energy = np.abs(grid.amplitude) ** 2

    # Extract peaks and bits
    peaks = compiler.extract_peaks(num_peaks=78)
    extractor = MetricExtractor(peaks, [])
    extracted_bits = extractor.peaks_to_key_bits()

    # Per-bit match: True where extracted matches target
    per_bit_match = [
        extracted_bits[i] == target_bits[i] for i in range(256)
    ]
    match_count = sum(per_bit_match)
    match_rate = match_count / 256

    return per_bit_match, match_rate, match_count, extracted_bits


def main():
    key = KeyInput(TARGET_KEY)
    target_bits = key.as_bits

    print(f"TARGET KEY: {key.as_hex}")
    print(f"GRID: {GRID_SIZE}x{GRID_SIZE}x{GRID_SIZE}")
    print(f"SWEEP: {FREQ_START}-{FREQ_END} MHz (step {FREQ_STEP})")
    print(f"PER-FREQ: {STEPS} steps, dt={DT}, strength={STRENGTH}")
    print()

    frequencies = list(range(FREQ_START, FREQ_END + 1, FREQ_STEP))
    num_freqs = len(frequencies)

    # bit_matrix[freq_idx][bit_idx] = True/False
    bit_matrix = np.zeros((num_freqs, 256), dtype=bool)
    freq_rates = []
    freq_counts = []
    all_extracted = []

    t0 = time.time()
    for fi, freq in enumerate(frequencies):
        per_bit, rate, count, extracted = run_at_frequency(key, freq, target_bits)
        bit_matrix[fi] = per_bit
        freq_rates.append(rate)
        freq_counts.append(count)
        all_extracted.append(extracted)

        elapsed = time.time() - t0
        eta = elapsed / (fi + 1) * (num_freqs - fi - 1)
        print(
            f"  freq={freq:3d} MHz: {rate:.4f} ({count:3d}/256) "
            f"[{fi+1}/{num_freqs}] {elapsed:.0f}s elapsed, ~{eta:.0f}s remaining"
        )

    total_time = time.time() - t0

    # -- Analysis --
    print()
    print("=" * 70)
    print(" FREQUENCY-TO-BIT ANALYSIS")
    print("=" * 70)

    # Best frequency overall
    best_freq_idx = np.argmax(freq_rates)
    print(f"  Best single frequency: {frequencies[best_freq_idx]} MHz "
          f"({freq_rates[best_freq_idx]:.4f}, {freq_counts[best_freq_idx]}/256)")

    # Top 10 frequencies
    top_indices = np.argsort(freq_rates)[::-1][:10]
    print(f"\n  Top 10 frequencies:")
    for idx in top_indices:
        print(f"    {frequencies[idx]:3d} MHz: {freq_rates[idx]:.4f} ({freq_counts[idx]}/256)")

    # Per-bit: best frequency
    best_freq_per_bit = []
    for bit in range(256):
        # Find all frequencies that got this bit right
        correct_freqs = [frequencies[fi] for fi in range(num_freqs) if bit_matrix[fi, bit]]
        best_freq_per_bit.append(correct_freqs)

    # Combined: pick the best frequency for each bit
    combined_correct = 0
    combined_bits = []
    bit_source_freq = []
    for bit in range(256):
        if any(bit_matrix[:, bit]):
            combined_correct += 1
            combined_bits.append(target_bits[bit])
            # Pick the frequency with highest overall rate among those that got this bit right
            correct_freqs_idx = [fi for fi in range(num_freqs) if bit_matrix[fi, bit]]
            best_fi = max(correct_freqs_idx, key=lambda fi: freq_rates[fi])
            bit_source_freq.append(frequencies[best_fi])
        else:
            # No frequency got this bit right -- mark as unknown
            combined_bits.append(1 - target_bits[bit])  # wrong
            bit_source_freq.append(0)

    combined_rate = combined_correct / 256
    print(f"\n  COMBINED (best-per-bit): {combined_rate:.4f} ({combined_correct}/256)")

    # How many bits does each frequency uniquely crack?
    print(f"\n  Per-bit coverage:")
    bits_cracked_by_n = [0] * (num_freqs + 1)
    for bit in range(256):
        n_correct = sum(bit_matrix[:, bit])
        if n_correct <= num_freqs:
            bits_cracked_by_n[n_correct] += 1
    print(f"    Bits cracked by 0 frequencies:   {bits_cracked_by_n[0]}")
    print(f"    Bits cracked by 1-10 frequencies: {sum(bits_cracked_by_n[1:11])}")
    print(f"    Bits cracked by 11-50 frequencies: {sum(bits_cracked_by_n[11:51])}")
    print(f"    Bits cracked by 51+ frequencies:  {sum(bits_cracked_by_n[51:])}")

    # Bit difficulty: which bits are hardest (fewest frequencies crack them)?
    bit_difficulty = [sum(bit_matrix[:, bit]) for bit in range(256)]
    hardest_bits = np.argsort(bit_difficulty)[:20]
    easiest_bits = np.argsort(bit_difficulty)[::-1][:20]

    print(f"\n  Hardest bits (fewest frequencies succeed):")
    for bit in hardest_bits:
        print(f"    Bit {bit:3d}: {bit_difficulty[bit]:3d}/{num_freqs} freqs, target={target_bits[bit]}")

    print(f"\n  Easiest bits (most frequencies succeed):")
    for bit in easiest_bits:
        print(f"    Bit {bit:3d}: {bit_difficulty[bit]:3d}/{num_freqs} freqs, target={target_bits[bit]}")

    # Frequency bands: which bit ranges does each band cover best?
    print(f"\n  Frequency band analysis (match rate per 64-bit block):")
    print(f"  {'Freq':>6s}  {'Bits 0-63':>10s}  {'Bits 64-127':>12s}  {'Bits 128-191':>13s}  {'Bits 192-255':>13s}  {'Overall':>8s}")
    for fi in top_indices[:20]:
        blocks = []
        for start in range(0, 256, 64):
            block_match = sum(bit_matrix[fi, start:start+64]) / 64
            blocks.append(block_match)
        print(
            f"  {frequencies[fi]:4d} MHz  "
            f"{blocks[0]:10.3f}  {blocks[1]:12.3f}  {blocks[2]:13.3f}  {blocks[3]:13.3f}  "
            f"{freq_rates[fi]:8.4f}"
        )

    # Combined best key
    print()
    print("=" * 70)
    print(" COMBINED EXTRACTED KEY (best frequency per bit)")
    print("=" * 70)
    combined_hex = format(int("".join(str(b) for b in combined_bits), 2), "064x")
    print(f"  Target:    {key.as_hex}")
    print(f"  Extracted: {combined_hex}")
    print(f"  Match:     {combined_correct}/256 ({combined_rate:.1%})")
    print()

    # Bit map
    match_map = "".join(
        "+" if combined_bits[i] == target_bits[i] else "." for i in range(256)
    )
    print(f"  Bit map (+ = match, . = miss):")
    for start in range(0, 256, 64):
        print(f"    [{start:3d}-{start+63:3d}] {match_map[start:start+64]}")

    # Source frequency per bit
    print(f"\n  Source frequency per bit (which freq cracked each bit):")
    for start in range(0, 256, 64):
        freqs_line = " ".join(f"{bit_source_freq[i]:3d}" for i in range(start, start+64))
        print(f"    [{start:3d}-{start+63:3d}]")
        # Print in groups of 16 for readability
        for g in range(0, 64, 16):
            chunk = " ".join(f"{bit_source_freq[start+g+j]:3d}" for j in range(16))
            print(f"      {chunk}")

    print(f"\n  Runtime: {total_time:.1f}s")

    # Export CSV
    desktop = os.path.expanduser("~/Desktop")
    csv_path = os.path.join(desktop, "freq_bit_map.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["freq_mhz", "match_rate", "match_count"] + [f"bit_{i}" for i in range(256)]
        writer.writerow(header)
        for fi, freq in enumerate(frequencies):
            row = [freq, f"{freq_rates[fi]:.4f}", freq_counts[fi]]
            row.extend([1 if bit_matrix[fi, b] else 0 for b in range(256)])
            writer.writerow(row)
    print(f"  CSV exported: {csv_path}")

    print("=" * 70)


if __name__ == "__main__":
    main()
