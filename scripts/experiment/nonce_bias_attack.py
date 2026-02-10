"""ECDSA Nonce Bias Attack (Bleichenbacher / Minerva Style).

Demonstrates how even TINY biases in ECDSA nonce generation can leak
the private key over many signatures. This is the #1 real-world
cryptographic attack vector against Bitcoin and other ECDSA systems.

Historical attacks:
  - 2010: Fail0verflow cracked Sony PS3 ECDSA (reused nonce k)
  - 2013: Android SecureRandom bug (biased nonces, Bitcoin theft)
  - 2019: Minerva (timing-based nonce bias in smart cards)
  - 2020: LadderLeak (1-bit bias from non-constant-time ladder)
  - Various: Multiple ECDSA implementations with biased nonces

Defense: RFC 6979 deterministic nonce generation (HMAC-based).

References:
  - Bleichenbacher: "On the generation of DSA one-time keys" (2001)
  - Brumley & Tuveri: "Remote Timing Attacks are Still Practical" (2011)
  - Jancar et al: "Minerva: The curse of ECDSA nonces" (2020)
  - De Micheli & Heninger: "Recovering cryptographic keys from
    partial information, by example" (2020)
"""

import csv
import math
import os
import secrets
import sys
import time


# ================================================================
# Small EC arithmetic
# ================================================================

class SmallEC:
    """Elliptic curve y^2 = x^3 + ax + b over F_p."""
    def __init__(self, p, a, b):
        self.p = p
        self.a = a
        self.b = b
        self._order = None
        self._gen = None

    @property
    def order(self):
        if self._order is None:
            self._enumerate()
        return self._order

    @property
    def generator(self):
        if self._gen is None:
            self._enumerate()
        return self._gen

    def _enumerate(self):
        pts = [None]
        p = self.p
        qr = {}
        for y in range(p):
            qr.setdefault((y * y) % p, []).append(y)
        for x in range(p):
            rhs = (x * x * x + self.a * x + self.b) % p
            if rhs in qr:
                for y in qr[rhs]:
                    pts.append((x, y))
        self._order = len(pts)
        # Find a generator (point of maximal order)
        if len(pts) > 1:
            for pt in pts[1:]:
                if self.multiply(pt, self._order) is None:
                    if self._order <= 2 or self.multiply(pt, self._order // 2 if self._order % 2 == 0 else self._order) is not None:
                        self._gen = pt
                        break
            if self._gen is None:
                self._gen = pts[1]

    def on_curve(self, P):
        if P is None:
            return True
        x, y = P
        return (y * y - x * x * x - self.a * x - self.b) % self.p == 0

    def add(self, P, Q):
        if P is None: return Q
        if Q is None: return P
        p = self.p
        x1, y1 = P
        x2, y2 = Q
        if x1 == x2 and y1 == (p - y2) % p:
            return None
        if P == Q:
            if y1 == 0: return None
            lam = (3 * x1 * x1 + self.a) * pow(2 * y1, p - 2, p) % p
        else:
            lam = (y2 - y1) * pow((x2 - x1) % p, p - 2, p) % p
        x3 = (lam * lam - x1 - x2) % p
        y3 = (lam * (x1 - x3) - y1) % p
        return (x3, y3)

    def multiply(self, P, k):
        if k < 0:
            P = (P[0], (self.p - P[1]) % self.p)
            k = -k
        if k == 0 or P is None: return None
        result = None
        addend = P
        while k:
            if k & 1:
                result = self.add(result, addend)
            addend = self.add(addend, addend)
            k >>= 1
        return result


# ================================================================
# ECDSA with configurable nonce generation
# ================================================================

def is_probable_prime_small(n):
    """Simple primality test for small numbers."""
    if n < 2: return False
    if n < 4: return True
    if n % 2 == 0 or n % 3 == 0: return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True


def ecdsa_sign(ec, G, n, privkey, msg_hash, nonce_gen="uniform"):
    """Sign with various nonce generation strategies.

    nonce_gen options:
        "uniform"      - Correct: uniform random in [1, n-1]
        "reused"       - Fatal: same nonce every time
        "biased_msb"   - Dangerous: MSB always 0 (half the range)
        "biased_lsb"   - Dangerous: LSB always 0 (even nonces only)
        "biased_Nbits" - Dangerous: top N bits known/fixed
        "short"        - Dangerous: nonce from smaller range
        "timing_leak"  - Dangerous: nonce length varies (Minerva-style)
    """
    for _attempt in range(1000):
        if nonce_gen == "uniform":
            k = secrets.randbelow(n - 1) + 1
        elif nonce_gen == "reused":
            k = max(1, n // 7)  # always the same per curve
        elif nonce_gen == "biased_msb":
            # Top bit always 0 -> nonce in [1, n//2]
            k = secrets.randbelow(n // 2 - 1) + 1
        elif nonce_gen == "biased_lsb":
            # Bottom bit always 0 -> even nonces
            k = (secrets.randbelow(n // 2 - 1) + 1) * 2
        elif isinstance(nonce_gen, tuple) and nonce_gen[0] == "biased_top":
            # Top `bits` bits are known (set to 0)
            bits = nonce_gen[1]
            bit_len = n.bit_length()
            mask = (1 << (bit_len - bits)) - 1
            k = (secrets.randbelow(mask) + 1) % n
            if k == 0: k = 1
        elif nonce_gen == "short":
            # Nonce from much smaller range
            k = secrets.randbelow(min(n - 1, 2**16)) + 1
        elif isinstance(nonce_gen, tuple) and nonce_gen[0] == "timing_leak":
            # Nonce is uniform but we leak its bit length
            k = secrets.randbelow(n - 1) + 1
            # The "leak" is k.bit_length() (observable via timing)
        else:
            k = secrets.randbelow(n - 1) + 1

        R = ec.multiply(G, k)
        if R is None:
            continue
        r = R[0] % n
        if r == 0:
            continue
        k_inv = pow(k, -1, n)
        s = (k_inv * (msg_hash + r * privkey)) % n
        if s == 0:
            continue
        return r, s, k
    raise ValueError(f"Could not generate valid ECDSA signature after 1000 attempts (n={n})")


def ecdsa_verify(ec, G, n, pubkey, msg_hash, r, s):
    """Verify ECDSA signature."""
    if r <= 0 or r >= n or s <= 0 or s >= n:
        return False
    s_inv = pow(s, -1, n)
    u1 = (msg_hash * s_inv) % n
    u2 = (r * s_inv) % n
    P = ec.add(ec.multiply(G, u1), ec.multiply(pubkey, u2))
    if P is None:
        return False
    return P[0] % n == r


# ================================================================
# Attack implementations
# ================================================================

def attack_reused_nonce(n, r1, s1, h1, r2, s2, h2):
    """Recover private key from two signatures with the same nonce.

    If k is reused:
        s1 = k^{-1}(h1 + r1*d) mod n
        s2 = k^{-1}(h2 + r2*d) mod n
    If r1 == r2 (same R point):
        s1 - s2 = k^{-1}(h1 - h2 + (r1-r2)*d) = k^{-1}(h1 - h2)
        k = (h1 - h2) / (s1 - s2) mod n
        d = (s1*k - h1) / r1 mod n
    """
    if r1 != r2:
        return None  # Different nonces
    ds = (s1 - s2) % n
    if ds == 0:
        return None
    dh = (h1 - h2) % n
    k = (dh * pow(ds, -1, n)) % n
    d = ((s1 * k - h1) * pow(r1, -1, n)) % n
    return d


def attack_biased_msb(ec, G, n, signatures, known_bits=1):
    """Recover private key when top `known_bits` of nonce are known (0).

    With top bit = 0, nonce k < n/2.
    Collect enough signatures and use lattice-style reduction.

    For our toy demo: brute force on the reduced search space.
    """
    # For small curves, brute force is tractable
    # In reality, this uses LLL lattice reduction (HNP)
    for r, s, h in signatures:
        # k is in [1, n // (2**known_bits)]
        max_k = n // (2 ** known_bits)
        for k_guess in range(1, max_k + 1):
            d = ((s * k_guess - h) * pow(r, -1, n)) % n
            # Verify: does this d produce the correct public key?
            if d > 0 and d < n:
                # Quick check: recompute r from k_guess
                R = ec.multiply(G, k_guess)
                if R is not None and R[0] % n == r:
                    return d
    return None


def attack_short_nonce(ec, G, n, r, s, h, max_k=65536):
    """Recover private key when nonce is from a small range.

    Brute force k in [1, max_k], then compute d.
    """
    for k_guess in range(1, min(max_k, n) + 1):
        R = ec.multiply(G, k_guess)
        if R is not None and R[0] % n == r:
            d = ((s * k_guess - h) * pow(r, -1, n)) % n
            return d, k_guess
    return None, None


# ================================================================
# Main
# ================================================================

def main():
    print()
    print("=" * 78)
    print("  ECDSA NONCE BIAS ATTACK")
    print("  How tiny biases in nonce generation break ECDSA")
    print("=" * 78)

    csv_rows = []

    # Use several small curves
    # ECDSA requires prime group order. Find small curves y^2=x^3+ax+b with prime order.
    candidate_curves = []
    print(f"  Searching for small curves with prime group order...")
    for p in [43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107,
              109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179,
              181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251,
              257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 503, 509, 521]:
        for a, b in [(0, 7), (0, 3), (1, 1), (2, 3), (3, 5)]:
            ec = SmallEC(p, a, b)
            n = ec.order
            if n > 20 and is_probable_prime_small(n):
                candidate_curves.append((p, a, b, f"F_{p}_a{a}b{b}", ec))
                if len(candidate_curves) >= 6:
                    break
        if len(candidate_curves) >= 6:
            break

    test_curves = []
    for p, a, b, label, ec in candidate_curves[:4]:
        test_curves.append((p, a, b, label))
        print(f"  {label}: order={ec.order} (PRIME) -- suitable for ECDSA")
    print()

    # ================================================================
    # PART 1: Nonce Reuse Attack (Sony PS3 / fail0verflow)
    # ================================================================
    print(f"\n  PART 1: Nonce Reuse Attack (Sony PS3 Style)")
    print(f"  {'='*70}")
    print(f"\n  If the same nonce k is used for two different messages,")
    print(f"  the private key can be recovered with simple algebra.")
    print(f"  This is what broke Sony's PS3 code signing in 2010.\n")

    for p, a, b, label in test_curves:
        ec = SmallEC(p, a, b)
        n = ec.order
        G = ec.generator
        if G is None or n <= 2:
            continue

        n_success = 0
        n_trials = 10
        total_time = 0

        for trial in range(n_trials):
            privkey = secrets.randbelow(n - 1) + 1
            pubkey = ec.multiply(G, privkey)

            h1 = secrets.randbelow(n - 1) + 1
            h2 = secrets.randbelow(n - 1) + 1

            try:
                r1, s1, k1 = ecdsa_sign(ec, G, n, privkey, h1, nonce_gen="reused")
                r2, s2, k2 = ecdsa_sign(ec, G, n, privkey, h2, nonce_gen="reused")
            except ValueError:
                continue

            t0 = time.time()
            recovered = attack_reused_nonce(n, r1, s1, h1, r2, s2, h2)
            dt = (time.time() - t0) * 1000

            if recovered == privkey:
                n_success += 1
            total_time += dt

        rate = n_success / n_trials
        print(f"  {label:8s}: {n_success}/{n_trials} keys recovered "
              f"({rate*100:.0f}%) avg {total_time/n_trials:.2f}ms")

        csv_rows.append({
            "curve": label, "order": n, "attack": "nonce_reuse",
            "bias_type": "k=constant", "sigs_needed": 2,
            "success_rate": rate, "avg_time_ms": round(total_time / n_trials, 3),
        })

    # ================================================================
    # PART 2: Biased MSB Attack (HNP-style)
    # ================================================================
    print(f"\n\n  PART 2: Biased MSB Attack (Hidden Number Problem)")
    print(f"  {'='*70}")
    print(f"\n  If the top bit(s) of the nonce are always 0,")
    print(f"  the private key leaks through lattice reduction.\n")

    for p, a, b, label in test_curves[:2]:  # Smaller curves for brute force
        ec = SmallEC(p, a, b)
        n = ec.order
        G = ec.generator
        if G is None or n <= 4:
            continue

        for known_bits in [1, 2, 4]:
            n_success = 0
            n_trials = 5
            total_time = 0

            for trial in range(n_trials):
                privkey = secrets.randbelow(n - 1) + 1
                pubkey = ec.multiply(G, privkey)

                sigs = []
                try:
                    for _ in range(3):
                        h = secrets.randbelow(n - 1) + 1
                        r, s, k = ecdsa_sign(ec, G, n, privkey, h,
                                              nonce_gen=("biased_top", known_bits))
                        sigs.append((r, s, h))
                except ValueError:
                    continue

                t0 = time.time()
                recovered = attack_biased_msb(ec, G, n, sigs, known_bits)
                dt = (time.time() - t0) * 1000

                if recovered == privkey:
                    n_success += 1
                total_time += dt

            rate = n_success / n_trials
            print(f"  {label:8s} ({known_bits} bits known): "
                  f"{n_success}/{n_trials} recovered ({rate*100:.0f}%) "
                  f"avg {total_time/n_trials:.1f}ms")

            csv_rows.append({
                "curve": label, "order": n, "attack": "biased_msb",
                "bias_type": f"top_{known_bits}_bits_zero",
                "sigs_needed": 3, "success_rate": rate,
                "avg_time_ms": round(total_time / n_trials, 2),
            })

    # ================================================================
    # PART 3: Short Nonce Attack
    # ================================================================
    print(f"\n\n  PART 3: Short Nonce Attack")
    print(f"  {'='*70}")
    print(f"\n  If nonces come from a small range (e.g., 16 bits),")
    print(f"  brute force on k recovers the key from ONE signature.\n")

    for p, a, b, label in test_curves:
        ec = SmallEC(p, a, b)
        n = ec.order
        G = ec.generator
        if G is None or n <= 2:
            continue

        n_success = 0
        n_trials = 5
        total_time = 0

        for trial in range(n_trials):
            privkey = secrets.randbelow(n - 1) + 1
            h = secrets.randbelow(n - 1) + 1
            try:
                r, s, k_actual = ecdsa_sign(ec, G, n, privkey, h, nonce_gen="short")
            except ValueError:
                continue

            t0 = time.time()
            recovered, k_found = attack_short_nonce(ec, G, n, r, s, h)
            dt = (time.time() - t0) * 1000

            if recovered == privkey:
                n_success += 1
            total_time += dt

        rate = n_success / n_trials
        print(f"  {label:8s}: {n_success}/{n_trials} recovered ({rate*100:.0f}%) "
              f"avg {total_time/n_trials:.1f}ms (1 signature)")

        csv_rows.append({
            "curve": label, "order": n, "attack": "short_nonce",
            "bias_type": "k<65536", "sigs_needed": 1,
            "success_rate": rate,
            "avg_time_ms": round(total_time / n_trials, 2),
        })

    # ================================================================
    # PART 4: Statistical Bias Detection
    # ================================================================
    print(f"\n\n  PART 4: Detecting Nonce Bias from Signatures")
    print(f"  {'='*70}")
    print(f"\n  Given only (r, s) pairs (public!), can we detect bias?\n")

    # Use the largest prime-order curve we found
    if test_curves:
        p, a, b, label = test_curves[-1]
    else:
        p, a, b, label = 67, 0, 7, "F_67_a0b7"
    ec = SmallEC(p, a, b)
    n = ec.order
    G = ec.generator
    print(f"  Using {label} (order {n}) for bias detection\n")

    if G and n > 2:
        privkey = secrets.randbelow(n - 1) + 1

        for nonce_type in ["uniform", "biased_msb", "biased_lsb"]:
            r_values = []
            for _ in range(200):
                h = secrets.randbelow(n - 1) + 1
                try:
                    r, s, k = ecdsa_sign(ec, G, n, privkey, h, nonce_gen=nonce_type)
                    r_values.append(r)
                except ValueError:
                    pass

            # Statistical tests on r values
            if not r_values:
                print(f"  {nonce_type:12s}: no valid signatures generated")
                continue
            mean_r = sum(r_values) / len(r_values)
            expected_mean = (n - 1) / 2
            var_r = sum((r - mean_r)**2 for r in r_values) / len(r_values)

            # r values for unbiased nonces should be ~uniform in [0, n-1]
            # Biased nonces create non-uniform r distribution
            half_count = sum(1 for r in r_values if r < n // 2)
            half_ratio = half_count / len(r_values)

            print(f"  {nonce_type:12s}: mean_r={mean_r:7.1f} (expected {expected_mean:.1f}), "
                  f"r<n/2: {half_ratio:.2%}, std={var_r**0.5:.1f}")

    # ================================================================
    # PART 5: Real-World Scaling Analysis
    # ================================================================
    print(f"\n\n  PART 5: Real-World Attack Scaling for secp256k1")
    print(f"  {'='*70}")

    secp256k1_n = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141

    print(f"""
  secp256k1 group order: n = 2^256 - 432420386565659656852420866390673177326

  NONCE REUSE (k = constant):
    Signatures needed: 2
    Computation: ~0 (simple algebra)
    Status: INSTANT BREAK. Defense: never reuse nonces.

  BIASED MSB (top b bits known/fixed):
    Uses lattice reduction (LLL/BKZ on the Hidden Number Problem)
    b=1 (half range): ~200 signatures needed (De Micheli & Heninger 2020)
    b=2 (quarter range): ~100 signatures needed
    b=8 (top byte known): ~30 signatures needed
    Computation: polynomial in n (lattice reduction is fast)
    Status: REAL THREAT. Minerva CVE-2019-15809 exploited exactly this.

  BIASED LSB (even nonces only):
    Equivalent to 1-bit MSB bias
    ~200 signatures needed
    Status: REAL THREAT. LadderLeak (2020) exploited this.

  SHORT NONCE (k < 2^t):
    Brute force: 2^t operations from 1 signature
    t=32: 4 billion ops (~seconds)
    t=64: ~10^19 ops (~years)
    t=128: 2^128 ops (infeasible)
    For full 256-bit nonces: equivalent to ECDLP (impossible)
    Status: REAL THREAT if nonce space is reduced.

  TIMING-BASED BIAS (Minerva):
    Non-constant-time scalar multiply leaks nonce bit length
    Bit length = floor(log2(k)) + 1 is observable via timing
    This gives ~8 bits of nonce information per signature
    With ~100 signatures: enough for lattice attack
    Status: REAL THREAT for non-constant-time implementations.

  DEFENSE: RFC 6979
    k = HMAC_DRBG(private_key, message_hash)
    Deterministic: same message always gets same k
    Uniform: HMAC output is indistinguishable from random
    No randomness needed: immune to bad RNG
    Bitcoin Core + libsecp256k1: uses RFC 6979.
    """)

    # ================================================================
    # PART 6: Historical Timeline
    # ================================================================
    print(f"\n  PART 6: Historical Nonce Bias Attacks")
    print(f"  {'='*70}")

    timeline = [
        ("2010", "Sony PS3", "Nonce reuse (k=constant)", "PS3 jailbreak, piracy"),
        ("2013", "Android Bitcoin", "Java SecureRandom bias", "Bitcoin wallet theft"),
        ("2014", "Various HSMs", "Biased nonce generation", "Key recovery from signatures"),
        ("2019", "Minerva", "Timing-based nonce bias", "Smartcard key extraction (CVE-2019-15809)"),
        ("2020", "LadderLeak", "1-bit nonce bias", "Key recovery from 1-bit leak"),
        ("2020", "Multiple", "Lattice attacks on biased ECDSA", "Academic + practical demonstrations"),
        ("Ongoing", "Blockchain", "Historical signature analysis", "Old wallets with weak nonce generation"),
    ]

    for year, target, method, impact in timeline:
        print(f"  {year:8s} | {target:20s} | {method:30s} | {impact}")

    # ================================================================
    # SUMMARY
    # ================================================================
    print(f"\n\n{'='*78}")
    print(f"  SUMMARY")
    print(f"{'='*78}")
    print(f"""
  Nonce bias is the most PRACTICAL attack on ECDSA:

  1. Nonce reuse = instant key recovery (algebra only)
  2. Biased nonces = key recovery via lattice reduction
     - Even 1 bit of bias is enough with ~200 signatures
     - Timing leaks give ~8 bits per signature
  3. Short nonces = brute force key recovery

  Why this matters for Bitcoin:
  - Every ECDSA signature exposes (r, s) publicly on the blockchain
  - If ANY signing produced biased nonces, keys are recoverable
  - Historical transactions are permanently public -- can't be undone
  - Old wallets with weak RNG or non-constant-time signing are at risk

  Defense stack (all used by Bitcoin Core):
  - RFC 6979 deterministic nonces (eliminates randomness dependency)
  - Constant-time scalar multiplication (eliminates timing leaks)
  - libsecp256k1 validates all intermediate values
  - Taproot (BIP 340): Schnorr signatures (simpler, less nonce-sensitive)

  Bottom line: Nonce bias is a REAL and ONGOING threat, but modern
  implementations (libsecp256k1 + RFC 6979) are immune. The danger
  is to OLD wallets and BAD implementations.
    """)
    print("=" * 78)

    # Write CSV
    desktop = os.path.expanduser("~/Desktop")
    csv_path = os.path.join(desktop, "nonce_bias_attack.csv")
    if csv_rows:
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
            w.writeheader()
            w.writerows(csv_rows)
        print(f"\n  CSV written to {csv_path}")


if __name__ == "__main__":
    main()
