"""Least-squares key cracker.

The encoding is: angular_field = sum(coeffs[i] * Y_i(theta, phi)) / max_val
where coeffs[i] = +1 or -1 based on bit value, and Y_i are the first 256
real spherical harmonics.

To crack: build the basis matrix A where A[j,i] = Y_i(theta_j, phi_j),
extract the shell at r=0.5, and solve A*x = shell via least squares.
The signs of x give the bits.

This is the exact algebraic inverse of the encoding.
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


def build_sh_basis_matrix(grid_size):
    """Build the 256-column SH basis matrix on the encoding grid.

    Uses the EXACT same theta/phi as KeyInput.to_grid_state().
    Returns A of shape (grid_size^2, 256).
    """
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

    return A, theta_grid, phi_grid


def lsq_extract(grid, A, grid_size=GRID_SIZE):
    """Extract 256 coefficients via least-squares on multiple shells."""
    # Try multiple radial shells and pick the most confident result
    r_mid = np.argmin(np.abs(grid.r_coords - 0.5))

    # Get the radial weights used in encoding
    r_encoding = np.linspace(0, 1, grid_size)
    radial = np.exp(-((r_encoding - 0.5) ** 2) / 0.1)

    # Use several shells around the peak for robustness
    shell_range = range(max(0, r_mid - 5), min(grid_size, r_mid + 6))
    all_coeffs = []

    for ir in shell_range:
        shell = grid.amplitude[ir, :, :]
        b = shell.ravel()

        # Solve least squares: A * x = b / radial_weight
        # Since grid[ir] = radial[ir] * angular_field, and angular_field = A * (coeffs/max_val)
        # We solve A * x = b, and x = coeffs * radial[ir] / max_val
        # The signs of x give us the bits regardless of the scalar
        if radial[ir] > 1e-10:
            x, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
            all_coeffs.append(x / radial[ir])

    # Average coefficients across shells (weighted by radial proximity to 0.5)
    if all_coeffs:
        avg_coeffs = np.mean(all_coeffs, axis=0)
    else:
        avg_coeffs = np.zeros(NUM_THREADS)

    bits = [1 if c > 0 else 0 for c in avg_coeffs]
    return bits, avg_coeffs


def score_bits(extracted, target):
    """Compare extracted bits to target."""
    matches = sum(a == b for a, b in zip(extracted, target))
    return matches, matches / 256


def main():
    key = KeyInput(TARGET_KEY)
    target_bits = key.as_bits

    print(f"TARGET KEY: {key.as_hex}")
    print(f"GRID: {GRID_SIZE}^3")
    print()

    t0 = time.time()

    # Build basis matrix (once)
    print("Building SH basis matrix (256 columns)...")
    A, theta_grid, phi_grid = build_sh_basis_matrix(GRID_SIZE)
    print(f"  Matrix shape: {A.shape}")
    print(f"  Condition number: {np.linalg.cond(A):.2f}")
    t1 = time.time()
    print(f"  Time: {t1-t0:.1f}s")
    print()

    best_rate = 0.0
    best_bits = []
    best_label = ""

    # ================================================================
    # Test 1: Direct LSQ on raw grid (no resonance)
    # ================================================================
    print("=" * 70)
    print("TEST 1: DIRECT LEAST-SQUARES (no resonance)")
    print("=" * 70)

    grid = SphericalVoxelGrid(size=GRID_SIZE)
    grid.initialize_from_key(key)

    bits, coeffs = lsq_extract(grid, A)
    matches, rate = score_bits(bits, target_bits)
    print(f"  Multi-shell LSQ: {rate:.4f} ({matches}/256)")

    # Show coefficient quality
    true_coeffs = 2.0 * np.array(target_bits, dtype=np.float64) - 1.0
    # Since the encoding normalizes, our coeffs are scaled versions of true_coeffs
    # Check sign agreement
    sign_match = np.sum(np.sign(coeffs) == true_coeffs)
    print(f"  Sign matches:    {sign_match}/256")

    # Confidence: |coeff| magnitude (higher = more certain)
    coeff_magnitudes = np.abs(coeffs)
    print(f"  Coeff magnitude: mean={np.mean(coeff_magnitudes):.6f}, "
          f"min={np.min(coeff_magnitudes):.6f}, max={np.max(coeff_magnitudes):.6f}")

    # Weak bits (low confidence)
    weak_threshold = np.percentile(coeff_magnitudes, 20)
    weak_bits = np.where(coeff_magnitudes < weak_threshold)[0]
    weak_correct = sum(bits[i] == target_bits[i] for i in weak_bits)
    print(f"  Weak bits (<20th pct, |c|<{weak_threshold:.6f}): "
          f"{len(weak_bits)} bits, {weak_correct}/{len(weak_bits)} correct")

    strong_bits = np.where(coeff_magnitudes >= weak_threshold)[0]
    strong_correct = sum(bits[i] == target_bits[i] for i in strong_bits)
    print(f"  Strong bits (>20th pct): "
          f"{len(strong_bits)} bits, {strong_correct}/{len(strong_bits)} correct")

    if rate > best_rate:
        best_rate = rate
        best_bits = bits[:]
        best_label = "Direct LSQ (no resonance)"

    # Single best shell
    r_mid = np.argmin(np.abs(grid.r_coords - 0.5))
    shell = grid.amplitude[r_mid, :, :].ravel()
    x, _, _, _ = np.linalg.lstsq(A, shell, rcond=None)
    single_bits = [1 if c > 0 else 0 for c in x]
    single_matches, single_rate = score_bits(single_bits, target_bits)
    print(f"  Single shell (r_mid={r_mid}): {single_rate:.4f} ({single_matches}/256)")
    if single_rate > best_rate:
        best_rate = single_rate
        best_bits = single_bits[:]
        best_label = "Single shell LSQ"

    print()

    # ================================================================
    # Test 2: LSQ after resonance at various settings
    # ================================================================
    print("=" * 70)
    print("TEST 2: LSQ AFTER RESONANCE")
    print("=" * 70)

    configs = [
        (78.0, 50, 0.01),
        (78.0, 100, 0.02),
        (78.0, 200, 0.05),
        (78.0, 500, 0.05),
        (78.0, 1000, 0.02),
        (39.0, 200, 0.05),
        (156.0, 200, 0.05),
        (1.0, 200, 0.05),
    ]
    for freq, steps, strength in configs:
        grid = SphericalVoxelGrid(size=GRID_SIZE)
        grid.initialize_from_key(key)

        _, tg, pg = np.meshgrid(
            grid.r_coords, grid.theta_coords, grid.phi_coords, indexing="ij"
        )
        for step in range(steps):
            t = step * 0.01
            vibration = np.sin(freq * pg + t) * np.cos(freq * tg)
            grid.amplitude *= 1.0 + vibration * strength
            grid.energy = np.abs(grid.amplitude) ** 2

        bits, coeffs = lsq_extract(grid, A)
        matches, rate = score_bits(bits, target_bits)
        print(f"  freq={freq:5.0f} steps={steps:4d} str={strength:.2f}: "
              f"{rate:.4f} ({matches}/256)")
        if rate > best_rate:
            best_rate = rate
            best_bits = bits[:]
            best_label = f"LSQ after {freq}MHz {steps}steps"

    print()

    # ================================================================
    # Test 3: Encoding grid match (use encoding theta/phi for readback)
    # ================================================================
    print("=" * 70)
    print("TEST 3: EXACT ENCODING GRID MATCH")
    print("=" * 70)

    # The encoding uses np.linspace(0, np.pi, grid_size) for theta
    # But the grid stores it at np.linspace(0.01, np.pi-0.01, grid_size)
    # Build a new basis on the GRID coordinates
    A_grid = np.zeros((GRID_SIZE * GRID_SIZE, NUM_THREADS), dtype=np.float64)
    tg2, pg2 = np.meshgrid(
        np.linspace(0.01, np.pi - 0.01, GRID_SIZE),
        np.linspace(0, 2 * np.pi, GRID_SIZE, endpoint=False),
        indexing="ij"
    )

    bit_idx = 0
    degree = 0
    while bit_idx < NUM_THREADS:
        for m in range(-degree, degree + 1):
            if bit_idx >= NUM_THREADS:
                break
            ylm = sph_harm_y(degree, m, tg2, pg2).real
            A_grid[:, bit_idx] = ylm.ravel()
            bit_idx += 1
        degree += 1

    grid = SphericalVoxelGrid(size=GRID_SIZE)
    grid.initialize_from_key(key)
    r_mid = np.argmin(np.abs(grid.r_coords - 0.5))
    shell = grid.amplitude[r_mid, :, :].ravel()

    # Now: shell values at grid coords, but they were computed at encoding coords
    # The amplitude at index (i,j) = angular_field_at_encoding_coords(i,j) * radial
    # So we should use the encoding-coord basis (A) not the grid-coord basis (A_grid)
    x_enc, _, _, _ = np.linalg.lstsq(A, shell, rcond=None)
    bits_enc = [1 if c > 0 else 0 for c in x_enc]
    matches_enc, rate_enc = score_bits(bits_enc, target_bits)
    print(f"  Encoding-coord basis: {rate_enc:.4f} ({matches_enc}/256)")

    x_grd, _, _, _ = np.linalg.lstsq(A_grid, shell, rcond=None)
    bits_grd = [1 if c > 0 else 0 for c in x_grd]
    matches_grd, rate_grd = score_bits(bits_grd, target_bits)
    print(f"  Grid-coord basis:     {rate_grd:.4f} ({matches_grd}/256)")

    if rate_enc > best_rate:
        best_rate = rate_enc
        best_bits = bits_enc[:]
        best_label = "Encoding-coord LSQ"
    if rate_grd > best_rate:
        best_rate = rate_grd
        best_bits = bits_grd[:]
        best_label = "Grid-coord LSQ"

    print()

    # ================================================================
    # Test 4: Oversampled readback
    # ================================================================
    print("=" * 70)
    print("TEST 4: OVERSAMPLED READBACK (156x156 angular grid)")
    print("=" * 70)

    oversample = 156  # 2x the grid size
    theta_os = np.linspace(0, np.pi, oversample)
    phi_os = np.linspace(0, 2 * np.pi, oversample)
    tg_os, pg_os = np.meshgrid(theta_os, phi_os, indexing="ij")

    A_os = np.zeros((oversample * oversample, NUM_THREADS), dtype=np.float64)
    bit_idx = 0
    degree = 0
    while bit_idx < NUM_THREADS:
        for m in range(-degree, degree + 1):
            if bit_idx >= NUM_THREADS:
                break
            ylm = sph_harm_y(degree, m, tg_os, pg_os).real
            A_os[:, bit_idx] = ylm.ravel()
            bit_idx += 1
        degree += 1

    # Interpolate the shell to 156x156
    from scipy.interpolate import RegularGridInterpolator

    grid = SphericalVoxelGrid(size=GRID_SIZE)
    grid.initialize_from_key(key)
    r_mid = np.argmin(np.abs(grid.r_coords - 0.5))
    shell_78 = grid.amplitude[r_mid, :, :]

    # Encoding coordinates for the 78x78 shell
    theta_enc = np.linspace(0, np.pi, GRID_SIZE)
    phi_enc = np.linspace(0, 2 * np.pi, GRID_SIZE)

    interp = RegularGridInterpolator(
        (theta_enc, phi_enc), shell_78,
        method="cubic", bounds_error=False, fill_value=0.0
    )

    # Evaluate at oversampled points
    pts = np.column_stack([tg_os.ravel(), pg_os.ravel()])
    shell_os = interp(pts)

    x_os, _, _, _ = np.linalg.lstsq(A_os, shell_os, rcond=None)
    bits_os = [1 if c > 0 else 0 for c in x_os]
    matches_os, rate_os = score_bits(bits_os, target_bits)
    print(f"  Oversampled (156x156): {rate_os:.4f} ({matches_os}/256)")

    if rate_os > best_rate:
        best_rate = rate_os
        best_bits = bits_os[:]
        best_label = "Oversampled LSQ (156x156)"

    # Try even higher
    for os_size in [234, 312]:
        theta_h = np.linspace(0, np.pi, os_size)
        phi_h = np.linspace(0, 2 * np.pi, os_size)
        tg_h, pg_h = np.meshgrid(theta_h, phi_h, indexing="ij")

        A_h = np.zeros((os_size * os_size, NUM_THREADS), dtype=np.float64)
        bit_idx = 0
        degree = 0
        while bit_idx < NUM_THREADS:
            for m in range(-degree, degree + 1):
                if bit_idx >= NUM_THREADS:
                    break
                ylm = sph_harm_y(degree, m, tg_h, pg_h).real
                A_h[:, bit_idx] = ylm.ravel()
                bit_idx += 1
            degree += 1

        pts_h = np.column_stack([tg_h.ravel(), pg_h.ravel()])
        shell_h = interp(pts_h)

        x_h, _, _, _ = np.linalg.lstsq(A_h, shell_h, rcond=None)
        bits_h = [1 if c > 0 else 0 for c in x_h]
        matches_h, rate_h = score_bits(bits_h, target_bits)
        print(f"  Oversampled ({os_size}x{os_size}): {rate_h:.4f} ({matches_h}/256)")

        if rate_h > best_rate:
            best_rate = rate_h
            best_bits = bits_h[:]
            best_label = f"Oversampled LSQ ({os_size}x{os_size})"

    print()

    # ================================================================
    # Test 5: Constrained recovery (we know coeffs are +/-1)
    # ================================================================
    print("=" * 70)
    print("TEST 5: CONSTRAINED RECOVERY (coeffs are +/- 1)")
    print("=" * 70)

    # Since we know the true coefficients are exactly +1 or -1 (before normalization),
    # we can use the LSQ solution's sign, but also verify by checking if forcing
    # each ambiguous bit to 0 or 1 reduces the residual
    grid = SphericalVoxelGrid(size=GRID_SIZE)
    grid.initialize_from_key(key)
    r_mid = np.argmin(np.abs(grid.r_coords - 0.5))
    shell = grid.amplitude[r_mid, :, :].ravel()

    x_lsq, _, _, _ = np.linalg.lstsq(A, shell, rcond=None)

    # Start with sign-based guess
    guess_coeffs = np.sign(x_lsq)
    guess_coeffs[guess_coeffs == 0] = 1.0

    # Find the scale factor: shell = scale * A * coeffs
    # scale = (shell . (A * coeffs)) / |A * coeffs|^2
    predicted = A @ guess_coeffs
    scale = np.dot(shell, predicted) / np.dot(predicted, predicted)

    residual = np.linalg.norm(shell - scale * predicted)
    print(f"  Initial residual: {residual:.6f}")

    # Greedy bit-flip: for each bit, try flipping and keep if residual improves
    improved = True
    iteration = 0
    while improved:
        improved = False
        iteration += 1
        flips = 0
        for i in range(NUM_THREADS):
            guess_coeffs[i] *= -1
            predicted_new = A @ guess_coeffs
            scale_new = np.dot(shell, predicted_new) / np.dot(predicted_new, predicted_new)
            residual_new = np.linalg.norm(shell - scale_new * predicted_new)
            if residual_new < residual:
                residual = residual_new
                scale = scale_new
                improved = True
                flips += 1
            else:
                guess_coeffs[i] *= -1  # flip back

        constrained_bits = [1 if c > 0 else 0 for c in guess_coeffs]
        matches, rate = score_bits(constrained_bits, target_bits)
        print(f"  Iteration {iteration}: residual={residual:.6f}, "
              f"flips={flips}, match={rate:.4f} ({matches}/256)")

    constrained_bits = [1 if c > 0 else 0 for c in guess_coeffs]
    matches, rate = score_bits(constrained_bits, target_bits)
    if rate > best_rate:
        best_rate = rate
        best_bits = constrained_bits[:]
        best_label = "Constrained greedy recovery"

    print()

    # ================================================================
    # FINAL REPORT
    # ================================================================
    total_time = time.time() - t0
    print("=" * 70)
    print(" LEAST-SQUARES CRACKING REPORT")
    print("=" * 70)
    print(f"  Target key:    {key.as_hex}")
    extracted_hex = format(int("".join(str(b) for b in best_bits), 2), "064x")
    print(f"  Extracted key: {extracted_hex}")
    print(f"  Best match:    {best_rate:.4f} ({int(best_rate*256)}/256 bits)")
    print(f"  Best strategy: {best_label}")
    print(f"  Runtime:       {total_time:.1f}s")
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
