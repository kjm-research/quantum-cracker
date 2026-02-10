"""Aggressive key cracking run.

Full 78x78x78 grid, multi-phase resonance ramping, SH filtering,
peak extraction at every phase. Tracks bit match convergence.
"""

import sys
import time

import numpy as np

sys.path.insert(0, "src")

from quantum_cracker.core.key_interface import KeyInput
from quantum_cracker.core.voxel_grid import SphericalVoxelGrid
from quantum_cracker.core.rip_engine import RipEngine
from quantum_cracker.core.harmonic_compiler import HarmonicCompiler
from quantum_cracker.analysis.metrics import MetricExtractor
from quantum_cracker.analysis.validation import Validator
from quantum_cracker.utils.types import SimulationConfig


TARGET_KEY = "06d88f2148757a251dd0ea0e6c4584e159a60cfd3f7217c7b0b111adec0efbca"
GRID_SIZE = 78


def extract_and_score(grid, key, compiler, engine, label, rip_history):
    """Extract peaks, score bits, print results."""
    peaks = compiler.extract_peaks(num_peaks=78)
    extractor = MetricExtractor(peaks, rip_history)
    extracted_bits = extractor.peaks_to_key_bits()
    validator = Validator(key, extracted_bits)

    match_rate = validator.bit_match_rate()
    matches = validator.bit_matches()
    ci = validator.confidence_interval()

    energy_sum = float(np.sum(grid.energy))
    energy_max = float(np.max(grid.energy))
    amp_range = float(np.max(grid.amplitude) - np.min(grid.amplitude))

    print(f"  [{label}]")
    print(f"    Bit match:   {match_rate:.4f} ({matches}/256)")
    print(f"    Confidence:  ({ci[0]:.3f}, {ci[1]:.3f})")
    print(f"    Energy sum:  {energy_sum:.2e}  max: {energy_max:.2e}")
    print(f"    Amp range:   {amp_range:.2e}")
    print(f"    Peaks: {len(peaks)}  top energy: {peaks[0].energy:.2e}" if peaks else "    No peaks")
    print()

    return match_rate, extracted_bits


def run():
    key = KeyInput(TARGET_KEY)
    print(f"TARGET KEY: {key.as_hex}")
    print(f"GRID: {GRID_SIZE}x{GRID_SIZE}x{GRID_SIZE} = {GRID_SIZE**3:,} voxels")
    print()

    # -- Initialize --
    t0 = time.time()
    print("=== INITIALIZING ===")
    grid = SphericalVoxelGrid(size=GRID_SIZE)
    grid.initialize_from_key(key)

    config = SimulationConfig(grid_size=GRID_SIZE, resonance_strength=0.05)
    compiler = HarmonicCompiler(grid, config=config)

    engine = RipEngine(config=config)
    engine.initialize_from_key(key)

    print(f"  Init time: {time.time()-t0:.1f}s")
    print(f"  Initial energy sum: {np.sum(grid.energy):.4f}")
    print(f"  Initial amp range: {np.max(grid.amplitude) - np.min(grid.amplitude):.4f}")
    print()

    best_rate = 0.0
    best_bits = []
    all_rates = []

    # -- Phase 1: Gentle warm-up --
    print("=" * 60)
    print("PHASE 1: WARM-UP  (500 steps, strength=0.02, dt=0.005)")
    print("=" * 60)
    compiler.config = SimulationConfig(grid_size=GRID_SIZE, resonance_strength=0.02)
    for step in range(500):
        compiler.time += 0.005
        compiler.apply_resonance(compiler.time)
        engine.step(dt=0.005)
        if (step + 1) % 100 == 0:
            rate, bits = extract_and_score(grid, key, compiler, engine, f"step {step+1}", engine.history)
            all_rates.append(rate)
            if rate > best_rate:
                best_rate = rate
                best_bits = bits[:]

    # -- Phase 2: Ramp resonance UP --
    print("=" * 60)
    print("PHASE 2: RAMP UP  (1000 steps, strength 0.05->0.20, dt=0.01)")
    print("=" * 60)
    for step in range(1000):
        # Linearly ramp strength from 0.05 to 0.20
        strength = 0.05 + 0.15 * (step / 1000.0)
        compiler.config = SimulationConfig(grid_size=GRID_SIZE, resonance_strength=strength)
        compiler.time += 0.01
        compiler.apply_resonance(compiler.time)
        engine.step(dt=0.01)
        if (step + 1) % 200 == 0:
            rate, bits = extract_and_score(grid, key, compiler, engine, f"step {step+1} str={strength:.3f}", engine.history)
            all_rates.append(rate)
            if rate > best_rate:
                best_rate = rate
                best_bits = bits[:]

    # -- Phase 3: SH Filter at l=78 --
    print("=" * 60)
    print("PHASE 3: SH FILTER  (isolate l=78 harmonic mode)")
    print("=" * 60)
    t3 = time.time()
    compiler.apply_spherical_harmonic_filter(l_target=78)
    print(f"  SH filter applied in {time.time()-t3:.1f}s")
    rate, bits = extract_and_score(grid, key, compiler, engine, "post-SH-filter", engine.history)
    all_rates.append(rate)
    if rate > best_rate:
        best_rate = rate
        best_bits = bits[:]

    # -- Phase 4: Post-filter resonance at high strength --
    print("=" * 60)
    print("PHASE 4: POST-FILTER BLAST  (1000 steps, strength=0.15, dt=0.01)")
    print("=" * 60)
    compiler.config = SimulationConfig(grid_size=GRID_SIZE, resonance_strength=0.15)
    for step in range(1000):
        compiler.time += 0.01
        compiler.apply_resonance(compiler.time)
        engine.step(dt=0.01)
        if (step + 1) % 200 == 0:
            rate, bits = extract_and_score(grid, key, compiler, engine, f"step {step+1}", engine.history)
            all_rates.append(rate)
            if rate > best_rate:
                best_rate = rate
                best_bits = bits[:]

    # -- Phase 5: Cool down with decreasing strength --
    print("=" * 60)
    print("PHASE 5: COOL DOWN  (500 steps, strength 0.15->0.01, dt=0.005)")
    print("=" * 60)
    for step in range(500):
        strength = 0.15 - 0.14 * (step / 500.0)
        compiler.config = SimulationConfig(grid_size=GRID_SIZE, resonance_strength=strength)
        compiler.time += 0.005
        compiler.apply_resonance(compiler.time)
        engine.step(dt=0.005)
        if (step + 1) % 100 == 0:
            rate, bits = extract_and_score(grid, key, compiler, engine, f"step {step+1} str={strength:.3f}", engine.history)
            all_rates.append(rate)
            if rate > best_rate:
                best_rate = rate
                best_bits = bits[:]

    # -- Phase 6: Second SH filter + fine resonance --
    print("=" * 60)
    print("PHASE 6: SECOND SH FILTER + FINE TUNING")
    print("=" * 60)
    t6 = time.time()
    compiler.apply_spherical_harmonic_filter(l_target=78)
    print(f"  SH filter applied in {time.time()-t6:.1f}s")
    rate, bits = extract_and_score(grid, key, compiler, engine, "post-SH-filter-2", engine.history)
    all_rates.append(rate)
    if rate > best_rate:
        best_rate = rate
        best_bits = bits[:]

    # Fine tuning: very small dt, moderate strength
    compiler.config = SimulationConfig(grid_size=GRID_SIZE, resonance_strength=0.08)
    for step in range(500):
        compiler.time += 0.002
        compiler.apply_resonance(compiler.time)
        engine.step(dt=0.002)
        if (step + 1) % 100 == 0:
            rate, bits = extract_and_score(grid, key, compiler, engine, f"fine step {step+1}", engine.history)
            all_rates.append(rate)
            if rate > best_rate:
                best_rate = rate
                best_bits = bits[:]

    # -- Phase 7: Multi-frequency sweep --
    print("=" * 60)
    print("PHASE 7: FREQUENCY SWEEP  (test nearby harmonics)")
    print("=" * 60)
    # Try slight frequency offsets to find resonance peak
    original_amplitude = grid.amplitude.copy()
    for freq_offset in [-2, -1, 0, 1, 2]:
        grid.amplitude = original_amplitude.copy()
        grid.energy = np.abs(grid.amplitude) ** 2
        freq = 78.0 + freq_offset
        for step in range(200):
            t = compiler.time + step * 0.01
            _, theta_grid, phi_grid = np.meshgrid(
                grid.r_coords, grid.theta_coords, grid.phi_coords, indexing="ij"
            )
            vibration = np.sin(freq * phi_grid + t) * np.cos(freq * theta_grid)
            grid.amplitude *= 1.0 + vibration * 0.10
            grid.energy = np.abs(grid.amplitude) ** 2

        peaks = compiler.extract_peaks(num_peaks=78)
        extractor = MetricExtractor(peaks, engine.history)
        extracted_bits = extractor.peaks_to_key_bits()
        validator = Validator(key, extracted_bits)
        rate = validator.bit_match_rate()
        matches = validator.bit_matches()
        print(f"  freq={freq:.0f} MHz: {rate:.4f} ({matches}/256)")
        all_rates.append(rate)
        if rate > best_rate:
            best_rate = rate
            best_bits = extracted_bits[:]

    # -- Phase 8: Eigenvalue analysis --
    print()
    print("=" * 60)
    print("PHASE 8: HAMILTONIAN ANALYSIS")
    print("=" * 60)
    # Restore best state for eigenvalue computation
    grid.amplitude = original_amplitude.copy()
    grid.energy = np.abs(grid.amplitude) ** 2
    compiler.config = SimulationConfig(grid_size=GRID_SIZE, resonance_strength=0.10)
    for step in range(500):
        compiler.time += 0.01
        compiler.apply_resonance(compiler.time)

    print("  Computing Hamiltonian eigenvalues...")
    eigenvalues = compiler.compute_hamiltonian_eigenvalues()
    print(f"  Ground state: {eigenvalues[0]:.4f}")
    print(f"  First excited: {eigenvalues[1]:.4f}")
    print(f"  Energy gap: {eigenvalues[1] - eigenvalues[0]:.4f}")
    print(f"  Eigenvalue spread: {eigenvalues[-1] - eigenvalues[0]:.4f}")

    # Final extraction
    rate, bits = extract_and_score(grid, key, compiler, engine, "FINAL", engine.history)
    all_rates.append(rate)
    if rate > best_rate:
        best_rate = rate
        best_bits = bits[:]

    # -- FINAL REPORT --
    total_time = time.time() - t0
    print()
    print("=" * 60)
    print(" CRACKING REPORT")
    print("=" * 60)
    print(f"  Target key:    {key.as_hex}")
    print(f"  Best match:    {best_rate:.4f} ({int(best_rate*256)}/256 bits)")
    print(f"  Total phases:  8")
    print(f"  Total steps:   3500+")
    print(f"  Rate history:  {' -> '.join(f'{r:.3f}' for r in all_rates)}")
    print(f"  Runtime:       {total_time:.1f}s")
    print()

    # Show extracted key
    if best_bits:
        extracted_hex = format(
            int("".join(str(b) for b in best_bits), 2), "064x"
        )
        print(f"  Extracted key: {extracted_hex}")
        print()
        # Bit-by-bit comparison
        target_bits = key.as_bits
        matches = sum(1 for a, b in zip(target_bits, best_bits) if a == b)
        mismatches = 256 - matches
        print(f"  Matches:       {matches}/256")
        print(f"  Mismatches:    {mismatches}/256")

        # Show which bit positions matched
        match_map = "".join(
            "+" if a == b else "." for a, b in zip(target_bits, best_bits)
        )
        print(f"  Bit map:       (+ = match, . = miss)")
        for i in range(0, 256, 64):
            print(f"    [{i:3d}-{i+63:3d}] {match_map[i:i+64]}")

    print()
    print(f"  Visible threads: {engine.num_visible}/256")
    print(f"  Final radius:    {engine.radius:.2e} m")
    print("=" * 60)


if __name__ == "__main__":
    run()
