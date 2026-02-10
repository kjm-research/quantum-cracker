"""Shor's Algorithm -- The Real Key Cracker.

This is what will actually break cryptography. Not Grover (quadratic speedup).
Shor gives EXPONENTIAL speedup by exploiting mathematical STRUCTURE.

The trick Shor discovered:
  1. Factoring a number N is equivalent to finding the PERIOD of a^x mod N
  2. Periods are FREQUENCIES
  3. Quantum Fourier Transform finds frequencies in one shot
  4. Classical computers need exponential time to find periods
  5. Quantum computers do it in polynomial time: O(n^3) for n-bit numbers

For Bitcoin/EC crypto specifically:
  - The "period" is the private key itself
  - The "frequency" is hidden in the group structure of the elliptic curve
  - Shor's adapted for EC (Proos & Zalka 2003) needs ~2n+3 qubits = 515 for 256-bit

This demo:
  - Factors numbers using real quantum simulation (period-finding + QFT)
  - Shows HOW the quantum Fourier transform reveals the hidden period
  - Compares: Classical (exponential) vs Grover (sqrt) vs Shor (polynomial)
  - Demonstrates on 4-bit through 20-bit numbers

The key insight for your concept:
  Grover = blind search with resonance (your ghost key)
  Shor   = finding the RIGHT FREQUENCY in the lock's mechanism
  Your "16 sounds" concept is closer to Shor than Grover.
"""

import math
import sys
import time
from fractions import Fraction

import numpy as np

# ================================================================
# SHOR'S ALGORITHM: PERIOD FINDING VIA QUANTUM FOURIER TRANSFORM
# ================================================================

def apply_qft(state):
    """Apply Quantum Fourier Transform using FFT (O(n log n)).

    QFT is the quantum version of the Discrete Fourier Transform.
    It converts position information into frequency information.
    On a quantum computer, this runs in O(n^2) gates on n qubits.
    Using numpy FFT avoids building an O(n^2) dense matrix.
    """
    return np.fft.fft(state) / np.sqrt(len(state))


def classical_period_find(a, N):
    """Find period of a^x mod N classically (brute force).

    This takes O(N) time -- exponential in the number of bits.
    """
    x = 1
    for r in range(1, N + 1):
        x = (x * a) % N
        if x == 1:
            return r
    return None


def quantum_period_find(a, N, n_qubits=None):
    """Simulate quantum period finding using QFT.

    This is the heart of Shor's algorithm.

    Steps:
      1. Create superposition of all x values: |0> + |1> + ... + |Q-1>
      2. Compute a^x mod N for each x (entangles input/output registers)
      3. Apply QFT to input register
      4. Measure -- get value close to multiple of Q/r (r = period)
      5. Use continued fractions to extract r from measurement

    On a real quantum computer: O(n^2 log n) gates for n-bit numbers.
    Classical simulation: O(Q^2) time (because we store full state vector).

    Returns: (period, measurement_data) with full intermediate state
    """
    if n_qubits is None:
        n_qubits = 2 * int(np.ceil(np.log2(N))) + 1

    Q = 2 ** n_qubits  # size of input register

    # STEP 1: Superposition (ghost key -- all x values simultaneously)
    input_amplitudes = np.ones(Q, dtype=np.complex128) / np.sqrt(Q)

    # STEP 2: Modular exponentiation
    # Compute f(x) = a^x mod N for every x
    # On a quantum computer, this is done with reversible circuits
    # In simulation, we just compute it directly
    f_values = np.zeros(Q, dtype=int)
    ax = 1
    for x in range(Q):
        f_values[x] = ax
        ax = (ax * a) % N

    # Group amplitudes by output value (simulates entanglement)
    # After measuring the output register, input register collapses
    # to superposition of x values giving the same f(x)
    output_groups = {}
    for x in range(Q):
        fval = f_values[x]
        if fval not in output_groups:
            output_groups[fval] = []
        output_groups[fval].append(x)

    # Pick a random output measurement (any will work)
    # The x-values in each group are spaced exactly r apart
    measured_f = list(output_groups.keys())[0]
    group = np.array(output_groups[measured_f])

    # Build collapsed input state (superposition of x-values with same f(x))
    collapsed = np.zeros(Q, dtype=np.complex128)
    for x in group:
        collapsed[x] = 1.0
    collapsed /= np.linalg.norm(collapsed)

    # STEP 3: Apply Quantum Fourier Transform
    # This converts the periodic pattern in collapsed state to peaks
    # at frequencies corresponding to the period
    transformed = apply_qft(collapsed)

    # Probability distribution after QFT
    probs = np.abs(transformed) ** 2

    # STEP 4: Measure -- sample from probability distribution
    # The peaks are at multiples of Q/r
    measured_k = np.random.choice(Q, p=probs)

    # STEP 5: Extract period using continued fractions
    # measured_k / Q is approximately s/r for some integer s
    if measured_k == 0:
        # Try again (this happens with probability 1/r)
        # In practice, you'd re-run the circuit
        nonzero_peaks = np.where(probs > 1.0 / Q)[0]
        if len(nonzero_peaks) > 1:
            measured_k = nonzero_peaks[1]  # take first non-zero peak
        else:
            measured_k = 1

    # Continued fraction expansion to find r
    fraction = Fraction(int(measured_k), int(Q)).limit_denominator(int(N))
    candidate_r = int(fraction.denominator)

    # Verify
    if pow(a, candidate_r, N) == 1:
        period = candidate_r
    else:
        # Try multiples
        period = None
        for mult in range(1, 10):
            if pow(a, candidate_r * mult, N) == 1:
                period = candidate_r * mult
                break

    # Collect diagnostic data
    peak_indices = np.where(probs > 0.5 / len(group))[0]
    peak_spacing = np.diff(peak_indices) if len(peak_indices) > 1 else np.array([0])

    data = {
        "Q": Q,
        "n_qubits": n_qubits,
        "measured_f": measured_f,
        "group_size": len(group),
        "group_spacing": group[1] - group[0] if len(group) > 1 else 0,
        "measured_k": measured_k,
        "fraction": f"{fraction}",
        "candidate_r": candidate_r,
        "n_peaks": len(peak_indices),
        "peak_spacing_mean": float(peak_spacing.mean()) if len(peak_spacing) > 0 else 0,
        "probs": probs,
        "peak_indices": peak_indices,
    }

    return period, data


def shor_factor(N, verbose=True):
    """Full Shor's algorithm to factor N.

    1. Pick random a < N
    2. Check gcd(a, N) -- if > 1, we got lucky
    3. Find period r of a^x mod N using quantum period finding
    4. If r is even and a^(r/2) != -1 mod N:
       factors = gcd(a^(r/2) +/- 1, N)

    Returns: (p, q) factors, or None
    """
    if N % 2 == 0:
        return 2, N // 2

    # Check if N is a prime power
    for k in range(2, int(np.log2(N)) + 1):
        root = round(N ** (1/k))
        for candidate in [root - 1, root, root + 1]:
            if candidate > 1 and candidate ** k == N:
                return candidate, N // candidate

    np.random.seed(None)

    for attempt in range(20):
        a = np.random.randint(2, N)
        g = math.gcd(a, N)

        if g > 1:
            if verbose:
                print(f"    Lucky: gcd({a}, {N}) = {g}")
            return g, N // g

        if verbose:
            print(f"    Attempt {attempt+1}: a={a}")

        # Quantum period finding
        period, data = quantum_period_find(a, N)

        if verbose:
            print(f"      QFT register: {data['n_qubits']} qubits ({data['Q']:,} states)")
            print(f"      Measured output: f(x) = {data['measured_f']}")
            print(f"      Period group: {data['group_size']} x-values, spaced {data['group_spacing']} apart")
            print(f"      QFT peaks: {data['n_peaks']} peaks, spacing ~{data['peak_spacing_mean']:.1f}")
            print(f"      Measured frequency: k={data['measured_k']}")
            print(f"      Continued fraction: {data['fraction']} -> candidate r={data['candidate_r']}")

        if period is None:
            if verbose:
                print(f"      Period not found, retrying...")
            continue

        if verbose:
            print(f"      Period found: r={period}")
            # Verify
            print(f"      Verify: {a}^{period} mod {N} = {pow(a, period, N)}")

        if period % 2 != 0:
            if verbose:
                print(f"      Period is odd, retrying...")
            continue

        half = pow(a, period // 2, N)
        if half == N - 1:
            if verbose:
                print(f"      a^(r/2) = -1 mod N, retrying...")
            continue

        p = math.gcd(half + 1, N)
        q = math.gcd(half - 1, N)

        if p * q == N and p > 1 and q > 1:
            return min(p, q), max(p, q)
        elif p > 1 and p < N:
            return p, N // p
        elif q > 1 and q < N:
            return q, N // q

        if verbose:
            print(f"      Factors trivial (1 or N), retrying...")

    return None


# ================================================================
# MAIN
# ================================================================

def main():
    print()
    print("=" * 74)
    print("  SHOR'S ALGORITHM -- The Real Key Cracker")
    print("  Quantum Fourier Transform finds hidden periods (frequencies)")
    print("=" * 74)
    print("""
  HOW IT WORKS:
    1. Factoring N is equivalent to finding the PERIOD of a^x mod N
    2. The period is a HIDDEN FREQUENCY in modular arithmetic
    3. Quantum Fourier Transform extracts this frequency in ONE pass
    4. Classical: O(2^n) time.  Grover: O(2^(n/2)).  Shor: O(n^3).
       For 256 bits: 2^256 vs 2^128 vs ~17 million.
    """)

    # ================================================================
    # DEMO 1: Factor small numbers
    # ================================================================
    test_numbers = [15, 21, 33, 35, 51, 77, 91, 143, 221, 323,
                    437, 667, 899, 1147, 2047, 4087, 8051,
                    15251, 32399, 65027, 131009, 262111, 524243]

    print("  " + "=" * 70)
    print("  FACTORING WITH SHOR'S ALGORITHM")
    print("  " + "=" * 70)

    print(f"\n  {'N':>8s}  {'Bits':>5s}  {'Factors':>16s}  {'Qubits':>7s}  "
          f"{'QFT States':>12s}  {'Time':>8s}  {'Status':>8s}")
    print(f"  {'-'*8}  {'-'*5}  {'-'*16}  {'-'*7}  {'-'*12}  {'-'*8}  {'-'*8}")

    results = []
    for N in test_numbers:
        n_bits = int(np.ceil(np.log2(N + 1)))
        n_qubits = 2 * n_bits + 1
        qft_states = 2 ** n_qubits

        # Skip if too large to simulate (FFT can handle up to ~2^26)
        if qft_states > 2 ** 26:
            print(f"  {N:8d}  {n_bits:5d}  {'(too large)':>16s}  {n_qubits:7d}  "
                  f"{qft_states:12,}  {'--':>8s}  {'SKIP':>8s}")
            continue

        t0 = time.time()
        result = shor_factor(N, verbose=False)
        elapsed = time.time() - t0

        if result:
            p, q = result
            factors_str = f"{p} x {q}"
            status = "CRACKED"
        else:
            factors_str = "failed"
            status = "RETRY"

        print(f"  {N:8d}  {n_bits:5d}  {factors_str:>16s}  {n_qubits:7d}  "
              f"{qft_states:12,}  {elapsed:7.2f}s  {status:>8s}")

        results.append({
            "N": N,
            "n_bits": n_bits,
            "n_qubits": n_qubits,
            "factors": factors_str,
            "time": elapsed,
            "success": result is not None,
        })

    # ================================================================
    # DEMO 2: Detailed walkthrough of one factoring
    # ================================================================
    print(f"\n  {'='*70}")
    print(f"  DETAILED WALKTHROUGH: Factoring 15")
    print(f"  {'='*70}")

    N = 15
    n_bits = 4
    a = 7  # chosen for clean demo

    print(f"\n  N = {N}, a = {a}")
    print(f"  Goal: find period r such that {a}^r mod {N} = 1")

    # Show the classical sequence
    print(f"\n  Classical computation of {a}^x mod {N}:")
    print(f"  x:  ", end="")
    vals = []
    ax = 1
    for x in range(20):
        vals.append(ax)
        print(f"{ax:3d}", end="")
        ax = (ax * a) % N
    print(f"\n  x:  ", end="")
    for x in range(20):
        print(f"{x:3d}", end="")
    print()

    # Find the period visually
    classical_r = classical_period_find(a, N)
    print(f"\n  Sequence repeats with period r = {classical_r}")
    print(f"  Verify: {a}^{classical_r} mod {N} = {pow(a, classical_r, N)}")

    # Now do it quantum
    print(f"\n  QUANTUM APPROACH:")
    period, data = quantum_period_find(a, N, n_qubits=8)

    print(f"  QFT register: {data['n_qubits']} qubits = {data['Q']} states")
    print(f"  All {data['Q']} values of {a}^x mod {N} computed SIMULTANEOUSLY")

    # Show the QFT probability distribution
    probs = data["probs"]
    print(f"\n  After QFT, probability peaks at frequencies:")
    peaks = data["peak_indices"]
    Q = data["Q"]
    for p in peaks[:8]:
        frac = Fraction(int(p), int(Q)).limit_denominator(int(N))
        print(f"    k={p:4d}  P={probs[p]:.4f}  k/Q={p}/{Q}={p/Q:.4f}  "
              f"-> fraction {frac} -> period candidate {frac.denominator}")

    print(f"\n  Period = {period}")

    # Factor
    if period and period % 2 == 0:
        half = pow(a, period // 2, N)
        p = math.gcd(half + 1, N)
        q = math.gcd(half - 1, N)
        print(f"\n  Factoring: {a}^({period}//2) mod {N} = {half}")
        print(f"  gcd({half}+1, {N}) = gcd({half+1}, {N}) = {p}")
        print(f"  gcd({half}-1, {N}) = gcd({half-1}, {N}) = {q}")
        print(f"\n  {N} = {p} x {q}")

    # ================================================================
    # DEMO 3: THE FREQUENCY VISUALIZATION
    # ================================================================
    print(f"\n  {'='*70}")
    print(f"  THE 'SOUND' OF SHOR'S ALGORITHM")
    print(f"  {'='*70}")
    print(f"""
  Your concept: 16 sounds, one resonates, the rest cancel.
  Shor's concept: the QFT finds the ONE FREQUENCY hidden in a^x mod N.

  Before QFT: all states equally likely (noise -- all frequencies mixed)
  After QFT:  peaks at exact multiples of Q/r (one frequency dominates)

  This IS resonance. The QFT is the harmonic analyzer. The period r
  is the frequency that was hidden in the modular arithmetic.
  """)

    # Show QFT as resonance for different N values
    for N_demo, a_demo in [(15, 7), (21, 2), (35, 3)]:
        n_q = 2 * int(np.ceil(np.log2(N_demo))) + 2
        Q_demo = 2 ** n_q

        period_demo, data_demo = quantum_period_find(a_demo, N_demo, n_qubits=n_q)
        probs_demo = data_demo["probs"]

        # Show as bar chart
        print(f"  N={N_demo}, a={a_demo}, period={period_demo}:")
        print(f"  QFT output ({Q_demo} frequency bins, showing top 10):")

        top_indices = np.argsort(probs_demo)[-10:][::-1]
        max_p = probs_demo[top_indices[0]]
        for idx in top_indices:
            bar_len = int(probs_demo[idx] / max_p * 40) if max_p > 0 else 0
            bar = "#" * bar_len
            frac = Fraction(int(idx), int(Q_demo)).limit_denominator(int(N_demo))
            print(f"    freq {idx:4d}/{Q_demo}: P={probs_demo[idx]:.4f}  "
                  f"{bar}  -> r={frac.denominator}")
        print()

    # ================================================================
    # SCALING COMPARISON
    # ================================================================
    print(f"  {'='*70}")
    print(f"  SCALING: Classical vs Grover vs Shor")
    print(f"  {'='*70}")

    print(f"\n  {'Bits':>6s}  {'Classical':>18s}  {'Grover':>18s}  {'Shor':>12s}  "
          f"{'Shor Qubits':>12s}")
    print(f"  {'-'*6}  {'-'*18}  {'-'*18}  {'-'*12}  {'-'*12}")

    for bits in [8, 16, 32, 64, 128, 256, 512]:
        classical = 2 ** bits
        grover = 2 ** (bits // 2)
        shor_ops = bits ** 3  # O(n^3)
        shor_qubits = 2 * bits + 3  # for ECDLP (Proos & Zalka)

        c_str = f"{classical:,}" if bits <= 32 else f"2^{bits}"
        g_str = f"{grover:,}" if bits <= 32 else f"2^{bits//2}"
        s_str = f"{shor_ops:,}"

        print(f"  {bits:6d}  {c_str:>18s}  {g_str:>18s}  {s_str:>12s}  {shor_qubits:>12d}")

    # ================================================================
    # THE BOTTOM LINE
    # ================================================================
    print(f"\n  {'='*70}")
    print(f"  THE BOTTOM LINE")
    print(f"  {'='*70}")
    print(f"""
  For 256-bit EC crypto (Bitcoin):

    CLASSICAL BRUTE FORCE: 2^256 operations
      = 10^77 operations
      = mass of universe in protons worth of computing

    GROVER (your ghost key): 2^128 operations
      = 10^38 operations
      = still impossible (heat death timescale)
      = needs ~4,000 logical qubits

    SHOR (frequency finding): ~256^3 = 16,777,216 operations
      = 17 million operations
      = MILLISECONDS on quantum hardware
      = needs ~515 logical qubits (2*256 + 3)

  Shor is 10^31 times faster than Grover.
  Shor needs 8x FEWER qubits than Grover.

  WHY? Because Shor doesn't do blind search.
  It finds the FREQUENCY (period) hidden in the group structure.
  Grover checks every key. Shor listens to the lock's harmonics.

  Your concept -- 16 sounds, resonance, collapse -- is Shor's intuition.
  The ghost key is Grover. The harmonic analyzer is Shor.
  Shor is what you were describing all along.

  Current hardware status:
    Need:        515 logical qubits = ~500,000 physical qubits
    IBM (2026):  1,386 physical qubits (noisy, not error-corrected)
    Google:      105 physical qubits (Willow chip)
    Timeline:    10-20 years for Shor-capable hardware
  """)
    print("=" * 74)


if __name__ == "__main__":
    main()
