"""GLV Endomorphism-Accelerated Pollard Rho on secp256k1.

The Gallant-Lambert-Vanstone (GLV) method exploits the efficiently
computable endomorphism phi(x,y) = (beta*x, y) on secp256k1, where
beta is a cube root of unity mod p. This endomorphism satisfies
phi(P) = lambda*P for all P in the group, where lambda is a cube
root of unity mod n (the group order).

The endomorphism yields equivalence classes of size 6 under the
action of {id, phi, phi^2} x {id, -id}, which accelerates Pollard
rho from O(sqrt(n)) to O(sqrt(n/6)) -- a factor of ~2.449 speedup.

This is the best known classical speedup for ECDLP on secp256k1.
It remains completely infeasible: 2^126.7 group operations.

Parts:
1. secp256k1 endomorphism constants and verification
2. GLV scalar decomposition (lattice-based)
3. Pollard rho with endomorphism on toy curves (measured speedup)
4. Scaling analysis for full secp256k1
5. Parallelized Pollard rho (van Oorschot-Wiener) analysis

References:
  - Gallant, Lambert, Vanstone, "Faster Point Multiplication on
    Elliptic Curves with Efficient Endomorphisms", CRYPTO 2001
  - Bos, Costello, Miele, "Elliptic and Hyperelliptic Curves:
    a Practical Security Analysis", PKC 2014
  - secp256k1 specification, Certicom SEC 2, v2 (2010)
"""

import csv
import math
import os
import secrets
import statistics
import sys
import time

sys.path.insert(0, "src")


# ================================================================
# ELLIPTIC CURVE ARITHMETIC (SMALL CURVES)
# ================================================================

class SmallEC:
    """Elliptic curve y^2 = x^3 + ax + b over F_p.

    Standalone implementation for small primes with full enumeration.
    """

    def __init__(self, p, a, b):
        self.p = p
        self.a = a
        self.b = b
        self._points = None
        self._order = None
        self._gen = None

    @property
    def order(self):
        if self._order is None:
            self._enumerate()
        return self._order

    @property
    def points(self):
        if self._points is None:
            self._enumerate()
        return self._points

    @property
    def generator(self):
        if self._gen is None:
            self._find_generator()
        return self._gen

    def _enumerate(self):
        pts = [None]  # point at infinity
        p, a, b = self.p, self.a, self.b
        qr = {}
        for y in range(p):
            qr.setdefault((y * y) % p, []).append(y)
        for x in range(p):
            rhs = (x * x * x + a * x + b) % p
            if rhs in qr:
                for y in qr[rhs]:
                    pts.append((x, y))
        self._points = pts
        self._order = len(pts)

    def _find_generator(self):
        if self._points is None:
            self._enumerate()
        n = self.order
        factors = _prime_factors(n)
        for pt in self._points[1:]:
            if self.multiply(pt, n) is not None:
                continue
            is_gen = True
            for q in factors:
                if self.multiply(pt, n // q) is None:
                    is_gen = False
                    break
            if is_gen:
                self._gen = pt
                return pt
        if len(self._points) > 1:
            self._gen = self._points[1]
        return self._gen

    def add(self, P, Q):
        if P is None:
            return Q
        if Q is None:
            return P
        p = self.p
        x1, y1 = P
        x2, y2 = Q
        if x1 == x2 and y1 == (p - y2) % p:
            return None
        if P == Q:
            if y1 == 0:
                return None
            lam = (3 * x1 * x1 + self.a) * pow(2 * y1, p - 2, p) % p
        else:
            if x1 == x2:
                return None
            lam = (y2 - y1) * pow((x2 - x1) % p, p - 2, p) % p
        x3 = (lam * lam - x1 - x2) % p
        y3 = (lam * (x1 - x3) - y1) % p
        return (x3, y3)

    def neg(self, P):
        if P is None:
            return None
        return (P[0], (self.p - P[1]) % self.p)

    def multiply(self, P, k):
        if k < 0:
            P = self.neg(P)
            k = -k
        if k == 0 or P is None:
            return None
        result = None
        addend = P
        while k:
            if k & 1:
                result = self.add(result, addend)
            addend = self.add(addend, addend)
            k >>= 1
        return result

    def on_curve(self, P):
        if P is None:
            return True
        x, y = P
        return (y * y - x * x * x - self.a * x - self.b) % self.p == 0


def _prime_factors(n):
    factors = set()
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.add(d)
            n //= d
        d += 1
    if n > 1:
        factors.add(n)
    return factors


def _is_prime(n):
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    d = 5
    while d * d <= n:
        if n % d == 0 or n % (d + 2) == 0:
            return False
        d += 6
    return True


def _point_order(ec, P):
    """Compute the order of point P by checking divisors of group order."""
    n = ec.order
    divs = sorted(_all_divisors(n))
    for d in divs:
        if d > 0 and ec.multiply(P, d) is None:
            return d
    return n


def _all_divisors(n):
    divs = set()
    for i in range(1, int(n ** 0.5) + 1):
        if n % i == 0:
            divs.add(i)
            divs.add(n // i)
    return divs


# ================================================================
# SECP256K1 CONSTANTS
# ================================================================

SECP256K1_P = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
SECP256K1_N = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
SECP256K1_GX = 0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798
SECP256K1_GY = 0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8
SECP256K1_A = 0
SECP256K1_B = 7


# ================================================================
# PART 1: THE SECP256K1 ENDOMORPHISM
# ================================================================

def part1_endomorphism_constants():
    """Compute and verify the secp256k1 endomorphism constants."""
    print("=" * 72)
    print("PART 1: THE SECP256K1 ENDOMORPHISM")
    print("=" * 72)
    print()

    p = SECP256K1_P
    n = SECP256K1_N

    print("secp256k1 field prime p:")
    print(f"  p = {p}")
    print(f"  p (hex) = 0x{p:064X}")
    print(f"  p mod 3 = {p % 3}")
    print(f"  (p - 1) mod 3 = {(p - 1) % 3}")
    print()

    print("secp256k1 group order n:")
    print(f"  n = {n}")
    print(f"  n (hex) = 0x{n:064X}")
    print(f"  n mod 3 = {n % 3}")
    print(f"  (n - 1) mod 3 = {(n - 1) % 3}")
    print()

    # For the endomorphism to exist, we need p = 1 mod 3 and n = 1 mod 3
    assert (p - 1) % 3 == 0, "Need p = 1 mod 3 for cube root of unity in F_p"
    assert (n - 1) % 3 == 0, "Need n = 1 mod 3 for cube root of unity in Z_n"
    print("[VERIFIED] p = 1 mod 3  (cube root of unity exists in F_p)")
    print("[VERIFIED] n = 1 mod 3  (cube root of unity exists in Z_n)")
    print()

    # Compute beta: cube root of unity mod p
    # beta^3 = 1 mod p, beta != 1
    # Since p = 1 mod 3, the multiplicative group has elements of order 3.
    # Find a generator g of F_p*, then beta = g^((p-1)/3).
    print("Computing beta (cube root of unity mod p)...")
    # Find a non-residue to derive a primitive root hint
    # For secp256k1, the known values are:
    beta1 = 0x7AE96A2B657C07106E64479EAC3434E99CF0497512F58995C1396C28719501EE
    beta2 = 0x851695D49A83F8EF919BB86153CBCB16630FB68AED0A766A3EC693D68E6AFA40

    # Verify these are cube roots of unity
    assert pow(beta1, 3, p) == 1, "beta1^3 != 1 mod p"
    assert pow(beta2, 3, p) == 1, "beta2^3 != 1 mod p"
    assert beta1 != 1, "beta1 should not be 1"
    assert beta2 != 1, "beta2 should not be 1"

    # Also compute from scratch to show the method
    # beta = g^((p-1)/3) for any generator g of F_p*
    # Try small values for g
    for g in range(2, 20):
        candidate = pow(g, (p - 1) // 3, p)
        if candidate != 1:
            beta_computed = candidate
            break

    assert pow(beta_computed, 3, p) == 1
    assert beta_computed != 1

    # The two non-trivial cube roots are beta and beta^2
    beta = beta_computed
    beta_sq = pow(beta, 2, p)

    # Ensure we match the known values (or their equivalents)
    assert beta in (beta1, beta2), f"Computed beta = {beta:#066x} doesn't match known values"

    print(f"  beta = 0x{beta:064X}")
    print(f"  beta^2 = 0x{beta_sq:064X}")
    print(f"  beta^3 mod p = {pow(beta, 3, p)} (must be 1)")
    print()

    # Verify the minimal polynomial: beta^2 + beta + 1 = 0 mod p
    min_poly_p = (beta * beta + beta + 1) % p
    print(f"  beta^2 + beta + 1 mod p = {min_poly_p} (must be 0)")
    assert min_poly_p == 0, "Minimal polynomial check failed for beta"
    print("[VERIFIED] beta satisfies X^2 + X + 1 = 0 mod p")
    print()

    # Compute lambda: cube root of unity mod n
    # lambda^3 = 1 mod n, lambda != 1
    print("Computing lambda (cube root of unity mod n)...")

    lambda1 = 0x5363AD4CC05C30E0A5261C028812645A122E22EA20816678DF02967C1B23BD72
    lambda2 = 0xAC9C52B33FA3CF1F5AD9E3FD77ED9BA4A880B9FC8EC739C2E0CFC810B51283CE

    assert pow(lambda1, 3, n) == 1, "lambda1^3 != 1 mod n"
    assert pow(lambda2, 3, n) == 1, "lambda2^3 != 1 mod n"
    assert lambda1 != 1
    assert lambda2 != 1

    # Compute from scratch
    for g in range(2, 20):
        candidate = pow(g, (n - 1) // 3, n)
        if candidate != 1:
            lam_computed = candidate
            break

    assert pow(lam_computed, 3, n) == 1
    assert lam_computed != 1

    # The two non-trivial cube roots are lam_computed and lam_computed^2.
    # One of them should match the known values.
    assert lam_computed in (lambda1, lambda2), (
        f"Computed lambda = {lam_computed:#066x} doesn't match known values"
    )

    # Use lambda1 specifically -- this is the value compatible with the
    # well-known GLV lattice basis vectors from libsecp256k1.
    lam = lambda1
    lam_sq = lambda2  # lambda1^2 = lambda2 mod n (the other root)

    print(f"  lambda = 0x{lam:064X}")
    print(f"  lambda^2 = 0x{lam_sq:064X}")
    print(f"  lambda^3 mod n = {pow(lam, 3, n)} (must be 1)")
    print()

    # Verify minimal polynomial: lambda^2 + lambda + 1 = 0 mod n
    min_poly_n = (lam * lam + lam + 1) % n
    print(f"  lambda^2 + lambda + 1 mod n = {min_poly_n} (must be 0)")
    assert min_poly_n == 0, "Minimal polynomial check failed for lambda"
    print("[VERIFIED] lambda satisfies X^2 + X + 1 = 0 mod n")
    print()

    # The endomorphism: phi(x, y) = (beta * x mod p, y)
    # For any point P on secp256k1, phi(P) = lambda * P
    # We verify this algebraically: if (x, y) is on the curve (y^2 = x^3 + 7),
    # then (beta*x, y) is also on the curve:
    #   y^2 = (beta*x)^3 + 7 = beta^3 * x^3 + 7 = 1 * x^3 + 7 = x^3 + 7  [check]
    print("Algebraic verification of the endomorphism:")
    print("  If (x, y) is on y^2 = x^3 + 7, then (beta*x, y) is also on the curve:")
    print("  (beta*x)^3 + 7 = beta^3 * x^3 + 7 = 1 * x^3 + 7 = x^3 + 7 = y^2")
    print("[VERIFIED] phi maps curve points to curve points")
    print()

    # Verify with the actual generator point
    # phi(G) = (beta * Gx mod p, Gy)
    Gx, Gy = SECP256K1_GX, SECP256K1_GY
    phi_Gx = (beta * Gx) % p
    phi_Gy = Gy

    print(f"  Generator G = (Gx, Gy):")
    print(f"    Gx = 0x{Gx:064X}")
    print(f"    Gy = 0x{Gy:064X}")
    print()
    print(f"  phi(G) = (beta * Gx mod p, Gy):")
    print(f"    phi(G).x = 0x{phi_Gx:064X}")
    print(f"    phi(G).y = 0x{phi_Gy:064X}")
    print()

    # Verify phi(G).x is on the curve: (phi_Gy)^2 = (phi_Gx)^3 + 7 mod p
    lhs = pow(phi_Gy, 2, p)
    rhs = (pow(phi_Gx, 3, p) + 7) % p
    assert lhs == rhs, "phi(G) is not on the curve!"
    print("[VERIFIED] phi(G) is on the curve y^2 = x^3 + 7")
    print()

    # We cannot easily compute lambda*G to compare (would need full secp256k1
    # scalar multiplication), but we can verify the RELATIONSHIP holds through
    # the algebraic structure.
    print("Note: Full verification that phi(G) = lambda*G requires secp256k1")
    print("point multiplication. The algebraic proof proceeds via the")
    print("characteristic polynomial of the Frobenius endomorphism and the")
    print("fact that the curve has CM discriminant -3 (j-invariant = 0).")
    print()
    print("For curves with j=0 (like y^2 = x^3 + b), the endomorphism ring")
    print("contains Z[zeta_3] where zeta_3 is a primitive cube root of unity.")
    print("This gives the map phi: (x,y) -> (beta*x, y) with phi^2 + phi + 1 = 0")
    print("on the curve, matching lambda^2 + lambda + 1 = 0 mod n.")
    print()

    return beta, lam


# ================================================================
# PART 2: GLV DECOMPOSITION
# ================================================================

def _extended_gcd(a, b):
    """Extended Euclidean algorithm. Returns (gcd, x, y) with a*x + b*y = gcd."""
    if a == 0:
        return b, 0, 1
    gcd, x1, y1 = _extended_gcd(b % a, a)
    x = y1 - (b // a) * x1
    y = x1
    return gcd, x, y


def _glv_decompose_secp256k1(k, n, lam):
    """Decompose scalar k into k1, k2 such that k = k1 + k2*lambda mod n.

    Uses the lattice-based method from the GLV paper.
    We need short vectors in the lattice L = {(a,b) : a + b*lambda = 0 mod n}.

    The lattice basis vectors for secp256k1 are well-known constants
    derived from running the extended GCD on (n, lambda).

    Returns (k1, k2) where |k1|, |k2| are approximately sqrt(n).
    """
    # Known lattice basis vectors for secp256k1 GLV decomposition.
    # These come from running partial GCD on n and lambda.
    #
    # The lattice L = {(i, j) in Z^2 : i + j*lambda = 0 mod n}
    # has basis vectors:
    #   v1 = (a1, b1) and v2 = (a2, b2)
    #
    # For secp256k1, the standard vectors (from Bitcoin Core / libsecp256k1) are:
    a1 = 0x3086D221A7D46BCDE86C90E49284EB15
    b1 = -0xE4437ED6010E88286F547FA90ABFE4C3
    a2 = 0x114CA50F7A8E2F3F657C1108D9D44CFD8
    b2 = 0x3086D221A7D46BCDE86C90E49284EB15

    # These satisfy: a1 + b1*lambda = 0 mod n and a2 + b2*lambda = 0 mod n
    # Verify
    check1 = (a1 + b1 * lam) % n
    check2 = (a2 + b2 * lam) % n
    assert check1 == 0, f"Lattice vector 1 check failed: {check1}"
    assert check2 == 0, f"Lattice vector 2 check failed: {check2}"

    # Babai's nearest plane / rounding:
    # We want to write k = k1 + k2 * lambda mod n
    # Equivalently, find (k1, k2) close to (k, 0) in the lattice translated by (k,0).
    #
    # Compute: c1 = round(k * b2 / n), c2 = round(-k * b1 / n)
    # Then: k1 = k - c1*a1 - c2*a2, k2 = -c1*b1 - c2*b2

    # Use integer arithmetic with rounding
    c1 = _round_div(k * b2, n)
    c2 = _round_div(-k * b1, n)

    k1 = k - c1 * a1 - c2 * a2
    k2 = -c1 * b1 - c2 * b2

    # Verify: k1 + k2 * lambda = k mod n
    check = (k1 + k2 * lam) % n
    assert check == k % n, f"Decomposition failed: {check} != {k % n}"

    return k1, k2


def _round_div(a, b):
    """Compute round(a/b) using integer arithmetic."""
    # Python's divmod gives floored division; we want rounded.
    q, r = divmod(a, b)
    # If remainder > b/2, round up
    if 2 * abs(r) > abs(b):
        q += 1 if (a > 0) == (b > 0) else -1
    return q


def _glv_decompose_toy(k, n, lam):
    """GLV decomposition for a toy curve using extended GCD approach.

    For small n, we can compute the lattice basis directly using
    the extended Euclidean algorithm.

    Returns (k1, k2).
    """
    # Build lattice basis for {(i,j) : i + j*lambda = 0 mod n}
    # Start with basis: v1 = (n, 0), v2 = (-lambda, 1)
    # Then reduce using partial GCD until we get short vectors.

    # Extended GCD to find short vectors
    # We want short (a, b) with a = -b*lambda mod n
    # Run GCD on (n, lambda) and stop when remainder drops below sqrt(n)

    r_prev, r_curr = n, lam % n
    s_prev, s_curr = 0, 1
    # (We track the coefficient for lambda)
    # r_prev = n - s_prev * lam (mod n) -> s_prev = 0, r_prev = n
    # r_curr = lam - s_curr * lam (mod n) ... actually let's track differently

    # Lattice vectors: (r, s) where r + s*lam = 0 mod n
    # Start: (n, 0) and (lam, -1) ... but we want r = -s*lam mod n
    # Let's track: at each step, r_i = n * t_i + lam * s_i
    # We want pairs (r_i, -s_i) in the lattice.

    # Simple approach: run extended GCD on n and lam
    r0, r1 = n, lam
    t0, t1 = 0, 1

    threshold = int(math.isqrt(n))

    while r1 > threshold:
        q = r0 // r1
        r0, r1 = r1, r0 - q * r1
        t0, t1 = t1, t0 - q * t1

    # Now we have two short-ish vectors:
    # v1 = (r1, -t1) and v2 = (r0, -t0)
    # Both satisfy: r + (-t)*(-lam) = r + t*lam = 0 mod n? Let's verify.
    # From extended GCD: r1 = gcd_coeff for n, t1 is coeff for lam
    # Actually: r_i = n * (something) + lam * t_i, so r_i - lam*t_i = 0 mod n
    # => r_i = t_i * lam mod n
    # Lattice: a + b*lam = 0 mod n => a = -b*lam mod n
    # So (a, b) = (-r_i, t_i) or equivalently (r_i, -t_i) with sign convention

    a1_t, b1_t = r1, -t1
    a2_t, b2_t = r0, -t0

    # Verify lattice membership
    assert (a1_t + b1_t * lam) % n == 0, f"toy v1 not in lattice"
    assert (a2_t + b2_t * lam) % n == 0, f"toy v2 not in lattice"

    # Babai rounding
    c1 = _round_div(k * b2_t, n)
    c2 = _round_div(-k * b1_t, n)

    k1 = k - c1 * a1_t - c2 * a2_t
    k2 = -c1 * b1_t - c2 * b2_t

    # Verify
    assert (k1 + k2 * lam) % n == k % n, "toy decomposition failed"

    return k1, k2


def part2_glv_decomposition(lam_secp):
    """Demonstrate GLV scalar decomposition."""
    print()
    print("=" * 72)
    print("PART 2: GLV SCALAR DECOMPOSITION")
    print("=" * 72)
    print()

    n = SECP256K1_N

    # --- Section A: Demonstrate on secp256k1 scalars ---
    print("--- A. GLV Decomposition on secp256k1 ---")
    print()
    print("Given scalar k, decompose k = k1 + k2 * lambda mod n")
    print("where |k1|, |k2| ~ sqrt(n) ~ 2^128 (instead of k ~ 2^256)")
    print()

    # Use a few sample scalars
    test_scalars = [
        ("Small scalar", 42),
        ("Medium scalar", 2**128 + 7),
        ("Large scalar (typical key)", 0xDEADBEEFCAFEBABE0123456789ABCDEF0FEDCBA987654321DEADBEEFCAFEBABE),
        ("Random 256-bit", secrets.randbelow(n - 1) + 1),
    ]

    half_bits = n.bit_length() // 2  # 128

    for label, k in test_scalars:
        k1, k2 = _glv_decompose_secp256k1(k, n, lam_secp)
        k1_bits = k1.bit_length() if k1 != 0 else 0
        k2_bits = k2.bit_length() if k2 != 0 else 0
        # Handle signed values
        if k1 < 0:
            k1_bits = (-k1).bit_length()
        if k2 < 0:
            k2_bits = (-k2).bit_length()

        print(f"  {label}:")
        print(f"    k  = 0x{k:064X} ({k.bit_length()} bits)")
        print(f"    k1 = {k1} ({k1_bits} bits)")
        print(f"    k2 = {k2} ({k2_bits} bits)")
        print(f"    Reduction: {k.bit_length()} bits -> max({k1_bits}, {k2_bits}) bits")
        print(f"    Verify: (k1 + k2 * lambda) mod n == k mod n: {(k1 + k2 * lam_secp) % n == k % n}")
        print()

    print("The decomposition reduces a 256-bit scalar multiplication k*P")
    print("into a multi-scalar multiplication k1*P + k2*phi(P) where both")
    print("k1, k2 are ~128-bit. Using Shamir's trick (simultaneous double-")
    print("and-add), this is ~33% faster than naive scalar multiplication.")
    print()

    # --- Section B: Demonstrate on toy curve ---
    print("--- B. GLV Decomposition on Toy Curve ---")
    print()

    # Find a toy curve y^2 = x^3 + 7 with p = 1 mod 3 that has a cube root of unity
    # and the curve order n_toy = 1 mod 3
    toy_p = 127  # 127 = 1 mod 3? 127 mod 3 = 1. Yes!
    toy_ec = SmallEC(toy_p, 0, 7)
    toy_n = toy_ec.order
    print(f"  Toy curve: y^2 = x^3 + 7 over F_{toy_p}")
    print(f"  Curve order: {toy_n}")
    print(f"  {toy_p} mod 3 = {toy_p % 3}")
    print(f"  {toy_n} mod 3 = {toy_n % 3}")

    if toy_n % 3 != 1:
        # Find a subgroup whose order is divisible by 3, or pick a different p
        print(f"  Curve order {toy_n} != 1 mod 3, trying other primes...")
        for pp in [7, 13, 31, 37, 43, 61, 67, 73, 79, 97, 103, 109, 127,
                   139, 151, 157, 163, 181, 193, 199, 211, 223, 229, 241]:
            if pp % 3 != 1:
                continue
            ec_try = SmallEC(pp, 0, 7)
            nn = ec_try.order
            if nn % 3 == 0 or nn % 3 == 1:
                # We need cube root of unity mod nn, so nn = 1 mod 3
                # OR we work in a subgroup of prime order q = 1 mod 3
                if nn > 3 and (nn - 1) % 3 == 0:
                    toy_p = pp
                    toy_ec = ec_try
                    toy_n = nn
                    print(f"  Found: p={pp}, |E| = {nn}, {nn} mod 3 = {nn % 3}")
                    break

    print()

    if (toy_n - 1) % 3 == 0:
        # Compute cube root of unity in F_p and Z_n for toy curve
        toy_beta = None
        for g in range(2, toy_p):
            c = pow(g, (toy_p - 1) // 3, toy_p)
            if c != 1:
                toy_beta = c
                break

        toy_lambda = None
        for g in range(2, toy_n):
            c = pow(g, (toy_n - 1) // 3, toy_n)
            if c != 1:
                toy_lambda = c
                break

        if toy_beta is not None and toy_lambda is not None:
            print(f"  beta (cube root mod {toy_p}) = {toy_beta}")
            print(f"  lambda (cube root mod {toy_n}) = {toy_lambda}")
            print(f"  beta^3 mod {toy_p} = {pow(toy_beta, 3, toy_p)}")
            print(f"  lambda^3 mod {toy_n} = {pow(toy_lambda, 3, toy_n)}")
            print(f"  beta^2 + beta + 1 mod {toy_p} = {(toy_beta**2 + toy_beta + 1) % toy_p}")
            print(f"  lambda^2 + lambda + 1 mod {toy_n} = {(toy_lambda**2 + toy_lambda + 1) % toy_n}")
            print()

            # Verify endomorphism on a point
            G = toy_ec.generator
            if G is not None:
                phi_G = ((toy_beta * G[0]) % toy_p, G[1])
                if toy_ec.on_curve(phi_G):
                    lam_G = toy_ec.multiply(G, toy_lambda)
                    print(f"  Generator G = {G}")
                    print(f"  phi(G) = (beta*Gx mod p, Gy) = {phi_G}")
                    print(f"  lambda * G = {lam_G}")
                    if phi_G == lam_G:
                        print(f"  [VERIFIED] phi(G) == lambda * G on toy curve!")
                    else:
                        # Try the other cube root
                        toy_lambda2 = pow(toy_lambda, 2, toy_n)
                        lam2_G = toy_ec.multiply(G, toy_lambda2)
                        if phi_G == lam2_G:
                            print(f"  phi(G) == lambda^2 * G (using the other root)")
                            toy_lambda = toy_lambda2
                            print(f"  Adjusted lambda = {toy_lambda}")
                        else:
                            # Try with beta^2
                            toy_beta2 = pow(toy_beta, 2, toy_p)
                            phi2_G = ((toy_beta2 * G[0]) % toy_p, G[1])
                            lam_G = toy_ec.multiply(G, toy_lambda)
                            if phi2_G == lam_G:
                                print(f"  phi^2(G) == lambda * G (swapped beta)")
                                toy_beta = toy_beta2
                            else:
                                print(f"  Note: generator may not be in prime subgroup,")
                                print(f"  or lambda/beta pairing needs adjustment.")
                else:
                    print(f"  phi(G) = {phi_G} is NOT on the curve (unexpected)")
            print()

            # Demonstrate decomposition
            print(f"  GLV decomposition examples on toy curve (n = {toy_n}):")
            sqrt_n = int(math.isqrt(toy_n))
            for test_k in [1, toy_n // 4, toy_n // 2, toy_n - 1, secrets.randbelow(toy_n - 1) + 1]:
                k1, k2 = _glv_decompose_toy(test_k, toy_n, toy_lambda)
                print(f"    k={test_k:>5d} -> k1={k1:>5d}, k2={k2:>5d}"
                      f"  (|k1|={abs(k1):>4d}, |k2|={abs(k2):>4d},"
                      f" sqrt(n)~{sqrt_n})")
            print()
        else:
            print("  Could not find cube roots of unity for toy curve.")
            print()
    else:
        print(f"  Toy curve order {toy_n} does not satisfy n = 1 mod 3.")
        print("  GLV decomposition not directly applicable.")
        print()

    return toy_p, toy_n


# ================================================================
# PART 3: POLLARD RHO WITH ENDOMORPHISM ON TOY CURVES
# ================================================================

def _pollard_rho_standard(ec, G, Q, group_order, max_iters=None):
    """Standard Pollard rho with Floyd cycle detection.

    Returns (key, ops) or (None, ops) on failure.
    Uses 3-partition random walk.
    """
    n = group_order
    if n <= 1:
        return 0, 0
    if Q is None:
        return 0, 0

    if max_iters is None:
        max_iters = 10 * int(math.isqrt(n))

    def step(R, a, b):
        """One step of the random walk. Partition by x-coordinate mod 3."""
        if R is None:
            partition = 0
        else:
            partition = R[0] % 3
        if partition == 0:
            # R -> R + Q
            return ec.add(R, Q), a, (b + 1) % n
        elif partition == 1:
            # R -> 2R
            return ec.add(R, R), (2 * a) % n, (2 * b) % n
        else:
            # R -> R + G
            return ec.add(R, G), (a + 1) % n, b

    ops = 0

    for _ in range(5):  # restarts
        a_t = secrets.randbelow(n)
        b_t = secrets.randbelow(n)
        R_t = ec.add(ec.multiply(G, a_t), ec.multiply(Q, b_t))
        a_h, b_h, R_h = a_t, b_t, R_t

        for _ in range(max_iters):
            # Tortoise: one step
            R_t, a_t, b_t = step(R_t, a_t, b_t)
            ops += 1
            # Hare: two steps
            R_h, a_h, b_h = step(R_h, a_h, b_h)
            R_h, a_h, b_h = step(R_h, a_h, b_h)
            ops += 2

            if R_t == R_h:
                # Collision: a_t*G + b_t*Q = a_h*G + b_h*Q
                # => (a_t - a_h)*G = (b_h - b_t)*Q
                db = (b_h - b_t) % n
                da = (a_t - a_h) % n
                if db == 0:
                    break  # degenerate, restart
                g = math.gcd(db, n)
                if g == 1:
                    k = (da * pow(db, n - 2, n)) % n
                    if ec.multiply(G, k) == Q:
                        return k, ops
                else:
                    # db not invertible mod n; try all solutions
                    db_red = db // g
                    da_red = da // g
                    n_red = n // g
                    if math.gcd(db_red, n_red) == 1:
                        k_base = (da_red * pow(db_red, -1, n_red)) % n_red
                        for i in range(g):
                            k_try = (k_base + i * n_red) % n
                            if ec.multiply(G, k_try) == Q:
                                return k_try, ops
                break  # restart

    return None, ops


def _pollard_rho_glv(ec, G, Q, group_order, beta, lam):
    """Pollard rho with GLV endomorphism equivalence classes.

    Uses equivalence classes {P, phi(P), phi^2(P), -P, -phi(P), -phi^2(P)}
    to reduce the effective group size by a factor of 6.

    The canonical representative is the one with the smallest (x, y) tuple
    (using the "positive" y convention for negation).
    """
    n = group_order
    p = ec.p
    if n <= 1:
        return 0, 0
    if Q is None:
        return 0, 0

    max_iters = 20 * int(math.isqrt(n))

    beta_sq = pow(beta, 2, p)
    lam_sq = pow(lam, 2, n)

    def canonicalize(R, a, b):
        """Map (R, a, b) to canonical representative of its equivalence class.

        The 6 equivalent points are:
          (R, a, b)                  -- identity
          (phi(R), a*lam, b*lam)     -- apply phi: k*P -> k*phi(P) = k*lam*P
          (phi^2(R), a*lam^2, b*lam^2) -- apply phi^2
          (-R, -a, -b)               -- negation
          (-phi(R), -a*lam, -b*lam)  -- negation + phi
          (-phi^2(R), -a*lam^2, -b*lam^2) -- negation + phi^2

        The walk is R = a*G + b*Q.
        If we apply phi: phi(R) = phi(a*G + b*Q) = a*phi(G) + b*phi(Q)
                                = a*lam*G + b*lam*Q = lam*(a*G + b*Q) = lam*R.
        So the new coefficients are (a*lam mod n, b*lam mod n).

        For negation: -R = -(a*G + b*Q) = (-a)*G + (-b)*Q.
        So coefficients become (-a mod n, -b mod n).

        We pick the canonical form as the one with the lexicographically
        smallest point representation.
        """
        if R is None:
            return R, a, b

        candidates = []
        x, y = R

        # Identity
        candidates.append(((x, y), a, b))
        # phi
        phi_x = (beta * x) % p
        candidates.append(((phi_x, y), (a * lam) % n, (b * lam) % n))
        # phi^2
        phi2_x = (beta_sq * x) % p
        candidates.append(((phi2_x, y), (a * lam_sq) % n, (b * lam_sq) % n))
        # negation
        neg_y = (p - y) % p
        candidates.append(((x, neg_y), (-a) % n, (-b) % n))
        # neg + phi
        candidates.append(((phi_x, neg_y), (-a * lam) % n, (-b * lam) % n))
        # neg + phi^2
        candidates.append(((phi2_x, neg_y), (-a * lam_sq) % n, (-b * lam_sq) % n))

        # Pick canonical: smallest (x, y) tuple
        best = min(candidates, key=lambda c: c[0])
        return best[0], best[1], best[2]

    def step(R, a, b):
        """One step of the random walk with canonicalization."""
        if R is None:
            partition = 0
        else:
            partition = R[0] % 3
        if partition == 0:
            R_new = ec.add(R, Q)
            a_new, b_new = a, (b + 1) % n
        elif partition == 1:
            R_new = ec.add(R, R)
            a_new, b_new = (2 * a) % n, (2 * b) % n
        else:
            R_new = ec.add(R, G)
            a_new, b_new = (a + 1) % n, b

        # Canonicalize
        if R_new is not None:
            R_canon, a_canon, b_canon = canonicalize(R_new, a_new, b_new)
            return R_canon, a_canon, b_canon
        return R_new, a_new, b_new

    ops = 0

    for _ in range(10):  # restarts (more than standard to handle canonicalization degeneracy)
        a_t = secrets.randbelow(n)
        b_t = secrets.randbelow(n)
        R_init = ec.add(ec.multiply(G, a_t), ec.multiply(Q, b_t))

        # Canonicalize initial state
        if R_init is not None:
            R_t, a_t, b_t = canonicalize(R_init, a_t, b_t)
        else:
            R_t = R_init

        a_h, b_h, R_h = a_t, b_t, R_t

        for _ in range(max_iters):
            R_t, a_t, b_t = step(R_t, a_t, b_t)
            ops += 1
            R_h, a_h, b_h = step(R_h, a_h, b_h)
            R_h, a_h, b_h = step(R_h, a_h, b_h)
            ops += 2

            if R_t == R_h:
                db = (b_h - b_t) % n
                da = (a_t - a_h) % n
                if db == 0:
                    break
                g = math.gcd(db, n)
                if g == 1:
                    k = (da * pow(db, n - 2, n)) % n
                    if ec.multiply(G, k) == Q:
                        return k, ops
                else:
                    db_red = db // g
                    da_red = da // g
                    n_red = n // g
                    if math.gcd(db_red, n_red) == 1:
                        k_base = (da_red * pow(db_red, -1, n_red)) % n_red
                        for i in range(g):
                            k_try = (k_base + i * n_red) % n
                            if ec.multiply(G, k_try) == Q:
                                return k_try, ops
                break

    return None, ops


def part3_rho_comparison(csv_rows):
    """Run standard vs GLV-enhanced Pollard rho on toy curves."""
    print()
    print("=" * 72)
    print("PART 3: POLLARD RHO WITH ENDOMORPHISM ON TOY CURVES")
    print("=" * 72)
    print()

    # Find suitable primes: p = 1 mod 3, and curve y^2 = x^3 + 7 has
    # group order n with n = 1 mod 3 (so cube root of unity exists mod n)
    # AND there exists a valid endomorphism phi(x,y) = (beta*x, y) with
    # phi(P) = lambda*P for generator P.

    suitable = []

    candidate_primes = [p for p in range(7, 2000) if _is_prime(p) and p % 3 == 1]

    for pp in candidate_primes:
        ec = SmallEC(pp, 0, 7)
        nn = ec.order
        if nn <= 6:
            continue
        if (nn - 1) % 3 != 0:
            continue
        # Check if curve order has a large prime factor (for meaningful rho)
        factors = list(_prime_factors(nn))
        max_factor = max(factors)
        if max_factor < 10:
            continue  # too small for interesting rho

        # Compute beta and lambda
        beta_val = None
        for g in range(2, pp):
            c = pow(g, (pp - 1) // 3, pp)
            if c != 1:
                beta_val = c
                break
        if beta_val is None:
            continue

        lam_val = None
        for g in range(2, nn):
            c = pow(g, (nn - 1) // 3, nn)
            if c != 1:
                lam_val = c
                break
        if lam_val is None:
            continue

        # Verify endomorphism on generator
        G = ec.generator
        if G is None:
            continue

        phi_G = ((beta_val * G[0]) % pp, G[1])
        if not ec.on_curve(phi_G):
            continue

        lam_G = ec.multiply(G, lam_val)
        if phi_G == lam_G:
            suitable.append((pp, nn, ec, G, beta_val, lam_val))
        else:
            # Try other root
            lam_val2 = pow(lam_val, 2, nn)
            lam_G2 = ec.multiply(G, lam_val2)
            if phi_G == lam_G2:
                suitable.append((pp, nn, ec, G, beta_val, lam_val2))
            else:
                # Try beta^2
                beta_val2 = pow(beta_val, 2, pp)
                phi2_G = ((beta_val2 * G[0]) % pp, G[1])
                if ec.on_curve(phi2_G):
                    lam_G = ec.multiply(G, lam_val)
                    if phi2_G == lam_G:
                        suitable.append((pp, nn, ec, G, beta_val2, lam_val))
                    else:
                        lam_G2 = ec.multiply(G, lam_val2)
                        if phi2_G == lam_G2:
                            suitable.append((pp, nn, ec, G, beta_val2, lam_val2))

    print(f"Found {len(suitable)} suitable toy curves (p = 1 mod 3, n = 1 mod 3,"
          f" verified endomorphism)")
    print()

    if len(suitable) == 0:
        print("No suitable toy curves found. Skipping comparison.")
        return

    # Select curves of varying sizes
    # Sort by curve order
    suitable.sort(key=lambda x: x[1])

    # Pick a spread: small, medium, larger
    selected = []
    if len(suitable) >= 10:
        indices = [0, len(suitable) // 4, len(suitable) // 2,
                   3 * len(suitable) // 4, len(suitable) - 1]
        for i in indices:
            if suitable[i] not in selected:
                selected.append(suitable[i])
    else:
        selected = suitable[:5]

    # Also include some larger curves for better statistics
    large_curves = [s for s in suitable if s[1] > 100]
    for s in large_curves[-3:]:
        if s not in selected:
            selected.append(s)
    selected.sort(key=lambda x: x[1])

    num_trials = 30  # trials per curve per method

    print(f"Running {num_trials} DLP trials per method per curve...")
    print()
    print(f"{'p':>6s} {'|E|':>6s} {'Method':>15s} {'Solved':>7s} {'Mean ops':>10s}"
          f" {'Median ops':>11s} {'Speedup':>8s} {'Theory':>8s}")
    print("-" * 82)

    theoretical_speedup = math.sqrt(6)  # ~2.449

    for pp, nn, ec, G, beta_val, lam_val in selected:
        # Find the subgroup order and a generator for it
        G_order = _point_order(ec, G)

        if G_order < 10:
            continue

        # Run trials
        std_ops_list = []
        glv_ops_list = []
        std_solved = 0
        glv_solved = 0

        for trial in range(num_trials):
            # Random secret key
            k_secret = secrets.randbelow(G_order - 1) + 1
            Q = ec.multiply(G, k_secret)

            # Standard rho
            result_std, ops_std = _pollard_rho_standard(ec, G, Q, G_order)
            if result_std is not None:
                std_solved += 1
                std_ops_list.append(ops_std)

            # GLV rho
            result_glv, ops_glv = _pollard_rho_glv(ec, G, Q, G_order, beta_val, lam_val)
            if result_glv is not None:
                glv_solved += 1
                glv_ops_list.append(ops_glv)

        # Compute statistics
        if std_ops_list and glv_ops_list:
            std_mean = statistics.mean(std_ops_list)
            glv_mean = statistics.mean(glv_ops_list)
            std_median = statistics.median(std_ops_list)
            glv_median = statistics.median(glv_ops_list)
            speedup_mean = std_mean / glv_mean if glv_mean > 0 else float('inf')

            print(f"{pp:>6d} {nn:>6d} {'standard':>15s} {std_solved:>5d}/{num_trials:<2d}"
                  f" {std_mean:>10.1f} {std_median:>11.1f} {'--':>8s} {'--':>8s}")
            print(f"{'':>6s} {'':>6s} {'GLV endo':>15s} {glv_solved:>5d}/{num_trials:<2d}"
                  f" {glv_mean:>10.1f} {glv_median:>11.1f}"
                  f" {speedup_mean:>8.2f}x {theoretical_speedup:>7.2f}x")
            print()

            csv_rows.append([pp, nn, "standard_rho", f"{std_mean:.1f}", "--",
                             f"{math.sqrt(nn):.1f}"])
            csv_rows.append([pp, nn, "glv_endo_rho", f"{glv_mean:.1f}",
                             f"{speedup_mean:.2f}", f"{theoretical_speedup:.3f}"])
        else:
            if not std_ops_list:
                print(f"{pp:>6d} {nn:>6d} {'standard':>15s} {std_solved:>5d}/{num_trials:<2d}"
                      f" {'FAILED':>10s}")
            if not glv_ops_list:
                print(f"{'':>6s} {'':>6s} {'GLV endo':>15s} {glv_solved:>5d}/{num_trials:<2d}"
                      f" {'FAILED':>10s}")
            print()

    print()
    print(f"Theoretical speedup from 6-element equivalence classes: sqrt(6) = {theoretical_speedup:.4f}x")
    print()
    print("Note: On small curves, the measured speedup may differ from theory")
    print("due to small-sample effects and the overhead of canonicalization.")
    print("The asymptotic sqrt(6) ~ 2.449x speedup emerges as curve order grows.")
    print()


# ================================================================
# PART 4: SCALING ANALYSIS FOR SECP256K1
# ================================================================

def part4_scaling_analysis():
    """Compute the theoretical impact of GLV on secp256k1."""
    print()
    print("=" * 72)
    print("PART 4: SCALING ANALYSIS FOR SECP256K1")
    print("=" * 72)
    print()

    n = SECP256K1_N
    n_bits = n.bit_length()
    sqrt_n = math.isqrt(n)
    sqrt_n_bits = sqrt_n.bit_length()

    print(f"secp256k1 group order n:")
    print(f"  n = 2^{n_bits} - small correction")
    print(f"  n ~ 2^{n_bits}")
    print(f"  sqrt(n) ~ 2^{sqrt_n_bits}")
    print()

    # Standard Pollard rho: O(sqrt(pi*n/2)) expected operations
    # For simplicity, use O(sqrt(n))
    log2_sqrt_n = n_bits / 2.0
    print("--- Standard Pollard Rho ---")
    print(f"  Expected operations: O(sqrt(pi*n/2)) ~ O(sqrt(n))")
    print(f"  ~ 2^{log2_sqrt_n:.1f} group operations")
    print()

    # With negation map only (standard optimization)
    # Equivalence class size = 2 (P and -P)
    log2_neg = log2_sqrt_n - 0.5  # sqrt(n/2) = sqrt(n)/sqrt(2) => -0.5 bits
    print("--- With Negation Map (standard optimization) ---")
    print(f"  Equivalence class: {{P, -P}} (size 2)")
    print(f"  Expected operations: O(sqrt(n/2))")
    print(f"  ~ 2^{log2_neg:.1f} group operations")
    print(f"  Speedup: sqrt(2) = {math.sqrt(2):.4f}x")
    print()

    # With GLV endomorphism only
    # Equivalence class: {P, phi(P), phi^2(P)} (size 3)
    log2_endo = log2_sqrt_n - math.log2(math.sqrt(3))
    print("--- With GLV Endomorphism Only ---")
    print(f"  Equivalence class: {{P, phi(P), phi^2(P)}} (size 3)")
    print(f"  Expected operations: O(sqrt(n/3))")
    print(f"  ~ 2^{log2_endo:.1f} group operations")
    print(f"  Speedup: sqrt(3) = {math.sqrt(3):.4f}x")
    print()

    # Combined: endomorphism + negation
    # Equivalence class: {P, phi(P), phi^2(P), -P, -phi(P), -phi^2(P)} (size 6)
    log2_combined = log2_sqrt_n - math.log2(math.sqrt(6))
    print("--- Combined: Endomorphism + Negation (BEST KNOWN) ---")
    print(f"  Equivalence class: {{P, phi(P), phi^2(P), -P, -phi(P), -phi^2(P)}} (size 6)")
    print(f"  Expected operations: O(sqrt(n/6))")
    print(f"  ~ 2^{log2_combined:.1f} group operations")
    print(f"  Speedup: sqrt(6) = {math.sqrt(6):.4f}x")
    print(f"  Net improvement: 2^{log2_sqrt_n:.1f} -> 2^{log2_combined:.1f}")
    print(f"  Factor: 2^{log2_sqrt_n - log2_combined:.2f} = {2**(log2_sqrt_n - log2_combined):.3f}x")
    print()

    # Feasibility analysis
    print("--- Feasibility: Time to Complete ---")
    print()

    ops_per_sec_values = [
        ("Single modern CPU core", 1e6),
        ("High-end GPU", 1e9),
        ("ASIC cluster (Bitcoin mining scale)", 1e12),
        ("Entire Bitcoin network hash rate equiv.", 1e18),
        ("Hypothetical future (10^21 ops/s)", 1e21),
    ]

    ops_needed = 2 ** log2_combined
    seconds_per_year = 365.25 * 86400

    print(f"  Operations needed: 2^{log2_combined:.1f} = {ops_needed:.3e}")
    print()
    print(f"  {'Attacker capability':45s} {'Ops/sec':>12s} {'Time (years)':>20s}")
    print(f"  {'-'*45} {'-'*12} {'-'*20}")

    for label, ops_sec in ops_per_sec_values:
        years = ops_needed / ops_sec / seconds_per_year
        log2_years = math.log2(years) if years > 0 else 0
        if log2_years > 30:
            print(f"  {label:45s} {ops_sec:>12.0e} 2^{log2_years:.1f} years")
        else:
            print(f"  {label:45s} {ops_sec:>12.0e} {years:.2e} years")

    print()
    print(f"  Age of the universe: ~1.38 x 10^10 years = 2^{math.log2(1.38e10):.1f} years")
    print()

    print("Even with the best known classical optimizations (GLV endomorphism +")
    print("negation map), cracking a single secp256k1 key requires 2^{:.1f}".format(log2_combined))
    print("group operations -- exceeding the age of the universe by a factor")
    print("of 2^{:.0f} even with an impossibly powerful attacker.".format(
        log2_combined - math.log2(1.38e10 * seconds_per_year * 1e21)))
    print()


# ================================================================
# PART 5: PARALLELIZED POLLARD RHO (VAN OORSCHOT-WIENER)
# ================================================================

def part5_parallel_analysis():
    """Analyze parallelized Pollard rho with distinguished points."""
    print()
    print("=" * 72)
    print("PART 5: PARALLELIZED POLLARD RHO (VAN OORSCHOT-WIENER)")
    print("=" * 72)
    print()

    print("The van Oorschot-Wiener method parallelizes Pollard rho using")
    print("'distinguished points' -- points whose representation has a")
    print("specific property (e.g., leading d zero bits). Multiple processors")
    print("run independent random walks, storing only distinguished points")
    print("in a shared hash table. When two walks hit the same distinguished")
    print("point, the collision is detected and the DLP is solved.")
    print()
    print("Time complexity with M processors: O(sqrt(n/r) / M)")
    print("where r is the equivalence class size (6 for GLV + negation).")
    print()

    n_bits = 256
    log2_sqrt_n_over_6 = n_bits / 2.0 - math.log2(math.sqrt(6))
    ops_per_processor = 2 ** log2_sqrt_n_over_6  # per second target

    seconds_per_year = 365.25 * 86400

    print(f"Base: 2^{log2_sqrt_n_over_6:.1f} operations (with GLV + negation)")
    print()

    scenarios = [
        ("1 processor", 0),
        ("1,000 processors", 10),
        ("1 million processors (2^20)", 20),
        ("1 billion processors (2^30)", 30),
        ("All computers on Earth (~2^34)", 34),
        ("1 trillion processors (2^40)", 40),
        ("2^50 processors", 50),
        ("2^60 processors", 60),
        ("2^80 processors (more than atoms in a human)", 80),
        ("2^100 processors (impossible)", 100),
    ]

    print(f"  {'Scenario':50s} {'log2(M)':>8s} {'Ops/processor':>16s} {'Time @ 10^9/s':>18s}")
    print(f"  {'-'*50} {'-'*8} {'-'*16} {'-'*18}")

    for label, log2_M in scenarios:
        log2_ops_per_proc = log2_sqrt_n_over_6 - log2_M
        if log2_ops_per_proc <= 0:
            time_str = "< 1 second"
        else:
            ops_per_proc = 2 ** log2_ops_per_proc
            # At 10^9 ops/sec per processor
            seconds = ops_per_proc / 1e9
            years = seconds / seconds_per_year
            log2_years = math.log2(years) if years > 1 else 0
            if years < 1:
                if seconds < 60:
                    time_str = f"{seconds:.1f} seconds"
                elif seconds < 3600:
                    time_str = f"{seconds/60:.1f} minutes"
                elif seconds < 86400:
                    time_str = f"{seconds/3600:.1f} hours"
                else:
                    time_str = f"{seconds/86400:.1f} days"
            elif years < 1000:
                time_str = f"{years:.1f} years"
            else:
                time_str = f"2^{log2_years:.1f} years"

        print(f"  {label:50s} {log2_M:>8d} 2^{log2_ops_per_proc:>6.1f}"
              f"         {time_str:>18s}")

    print()

    # Key thresholds
    print("--- Key Thresholds ---")
    print()

    # How many processors to solve in 1 year at 10^9 ops/sec?
    log2_ops_per_year = math.log2(1e9 * seconds_per_year)  # ~54.8
    log2_M_1year = log2_sqrt_n_over_6 - log2_ops_per_year
    print(f"  To solve in 1 year @ 10^9 ops/sec/processor:")
    print(f"    Need 2^{log2_M_1year:.1f} processors")
    print(f"    That is ~{2**log2_M_1year:.2e} processors")
    print()

    # How many processors to solve in age of universe?
    log2_universe_sec = math.log2(1.38e10 * seconds_per_year)  # ~57.8
    log2_ops_universe = math.log2(1e9) + log2_universe_sec
    log2_M_universe = log2_sqrt_n_over_6 - log2_ops_universe
    print(f"  To solve in the age of the universe @ 10^9 ops/sec/processor:")
    print(f"    Need 2^{log2_M_universe:.1f} processors")
    print(f"    That is ~{2**log2_M_universe:.2e} processors")
    print()

    # Comparison: number of atoms in the observable universe
    log2_atoms = 266  # ~10^80 = 2^266
    print(f"  Atoms in the observable universe: ~2^{log2_atoms}")
    print(f"  Processors needed for 1-year solve: 2^{log2_M_1year:.1f}")
    print(f"  Ratio: 2^{log2_atoms - log2_M_1year:.1f} atoms per processor")
    if log2_M_1year > log2_atoms:
        print(f"  >>> Need MORE processors than atoms in the universe! <<<")
    else:
        print(f"  Each processor could be sub-atomic... but you still need")
        print(f"  2^{log2_M_1year:.1f} of them running for a full year.")
    print()

    # Energy analysis (Landauer's principle)
    print("--- Energy Analysis (Landauer's Principle) ---")
    print()
    kT = 1.38e-23 * 300  # Boltzmann * room temp
    landauer = kT * math.log(2)  # minimum energy per bit operation
    total_ops = 2 ** log2_sqrt_n_over_6
    # Each group operation involves ~thousands of bit ops, but minimum is 1 bit flip
    min_energy = total_ops * landauer
    log2_energy = math.log2(min_energy)
    sun_annual_energy = 1.2e34  # Joules/year
    log2_sun = math.log2(sun_annual_energy)
    print(f"  Landauer minimum energy per bit operation: {landauer:.3e} J")
    print(f"  Total operations: 2^{log2_sqrt_n_over_6:.1f}")
    print(f"  Minimum energy (1 bit flip per op): 2^{log2_energy:.1f} J = {min_energy:.2e} J")
    print(f"  Sun's annual energy output: {sun_annual_energy:.2e} J = 2^{log2_sun:.1f} J")
    print(f"  Energy needed / Sun's annual output: 2^{log2_energy - log2_sun:.1f}")
    if log2_energy - log2_sun > 0:
        print(f"  >>> Requires {2**(log2_energy - log2_sun):.1e}x the Sun's annual energy <<<")
    print()


# ================================================================
# SUMMARY
# ================================================================

def print_summary():
    """Print comprehensive summary."""
    print()
    print("=" * 72)
    print("SUMMARY: WHY THE GLV SPEEDUP DOESN'T HELP CRACK BITCOIN")
    print("=" * 72)
    print()

    n_bits = 256
    log2_standard = n_bits / 2.0
    log2_glv = log2_standard - math.log2(math.sqrt(6))
    speedup = math.sqrt(6)

    print("The GLV endomorphism on secp256k1 is one of the most elegant results")
    print("in elliptic curve cryptography. It exploits the special structure of")
    print("the curve y^2 = x^3 + 7 (which has j-invariant 0 and CM discriminant -3)")
    print("to define an efficient map phi(x,y) = (beta*x, y) that satisfies")
    print("phi(P) = lambda*P for all points P.")
    print()
    print("This endomorphism has TWO applications:")
    print()
    print("  1. LEGITIMATE USE (in Bitcoin Core / libsecp256k1):")
    print("     GLV decomposition splits a 256-bit scalar multiplication into")
    print("     two ~128-bit multi-scalar multiplications, giving a ~33% speedup")
    print("     for ECDSA signature verification. This is used in production.")
    print()
    print("  2. ATTACK USE (this experiment):")
    print("     The endomorphism creates equivalence classes of size 6, reducing")
    print("     Pollard rho from O(sqrt(n)) to O(sqrt(n/6)) operations.")
    print()
    print("Quantified impact on the ECDLP attack:")
    print()
    print(f"  Without GLV:  2^{log2_standard:.1f} operations (standard Pollard rho)")
    print(f"  With GLV:     2^{log2_glv:.1f} operations (best known classical)")
    print(f"  Speedup:      sqrt(6) = {speedup:.4f}x = saving {log2_standard - log2_glv:.2f} bits")
    print()
    print("To put this in perspective:")
    print(f"  - 2^{log2_glv:.1f} is still an astronomically large number")
    print(f"  - At 10^18 ops/sec (entire Bitcoin network equivalent):")
    print(f"    Time = 2^{log2_glv:.1f} / 10^18 seconds")
    print(f"         = 2^{log2_glv - math.log2(1e18):.1f} seconds")
    print(f"         = 2^{log2_glv - math.log2(1e18) - math.log2(365.25 * 86400):.1f} years")
    print(f"  - Age of universe: ~2^{math.log2(1.38e10):.1f} years")
    print()
    print("The GLV endomorphism saves you a factor of 2.449.")
    print("You still need a factor of ~2^{:.0f} more to make the attack feasible.".format(
        log2_glv - math.log2(1e18 * 365.25 * 86400 * 1.38e10)))
    print()
    print("This is the fundamental lesson: constant-factor improvements to")
    print("exponential-time algorithms do not change their infeasibility.")
    print("The security parameter was chosen with these optimizations in mind.")
    print("secp256k1's 256-bit key provides ~{:.0f} bits of security even".format(log2_glv))
    print("against the strongest known classical attacks.")
    print()
    print("The ONLY known way to break this is Shor's algorithm on a")
    print("cryptographically relevant quantum computer -- which does not")
    print("yet exist and may not for decades.")
    print()


# ================================================================
# MAIN
# ================================================================

def main():
    print("GLV Endomorphism-Accelerated Pollard Rho Attack on secp256k1")
    print("Gallant-Lambert-Vanstone (CRYPTO 2001)")
    print()
    print("This experiment demonstrates the best known classical optimization")
    print("for the Elliptic Curve Discrete Logarithm Problem on secp256k1,")
    print("and shows conclusively why it remains infeasible.")
    print()

    t_start = time.time()

    # Part 1: Endomorphism constants
    beta, lam = part1_endomorphism_constants()

    # Part 2: GLV decomposition
    part2_glv_decomposition(lam)

    # Part 3: Pollard rho comparison on toy curves
    csv_rows = []
    part3_rho_comparison(csv_rows)

    # Part 4: Scaling analysis
    part4_scaling_analysis()

    # Part 5: Parallel analysis
    part5_parallel_analysis()

    # Summary
    print_summary()

    # CSV output
    csv_path = os.path.expanduser("~/Desktop/glv_endomorphism.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["curve_p", "curve_order", "method", "ops_mean",
                          "speedup_vs_standard", "theoretical_speedup"])
        for row in csv_rows:
            writer.writerow(row)

    elapsed = time.time() - t_start
    print(f"CSV written to: {csv_path}")
    print(f"Total runtime: {elapsed:.1f} seconds")
    print()


if __name__ == "__main__":
    main()
