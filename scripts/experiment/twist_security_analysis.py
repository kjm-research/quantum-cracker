"""Quadratic Twist Security Analysis for secp256k1 and Related Curves.

Every elliptic curve E over F_p has a quadratic twist E': the set of points
(x, y) satisfying y^2 = x^3 + a*d^2*x + b*d^3 for some non-square d in F_p.
The twist is another elliptic curve over the same field, and its order
satisfies |E| + |E'| = 2p + 2 (from the Hasse bound / Weil conjectures).

If |E'| has small prime factors, an attacker who sends an x-coordinate where
x^3 + ax + b is NOT a quadratic residue mod p forces computation on the twist.
Without validation, scalar multiplication leaks private key bits modulo those
small factors -- a variant of the invalid curve / small subgroup attack.

Twist security is critical for:
  - X-only protocols (X25519, Taproot BIP 340/341)
  - Montgomery ladder implementations
  - Any system that processes bare x-coordinates

This script analyzes twist security across small demonstration curves and
the major standard curves used in practice.

References:
  - Bernstein & Lange, "SafeCurves: choosing safe curves for ECC"
    https://safecurves.cr.yp.to/twist.html
  - Bernstein, "Curve25519: new Diffie-Hellman speed records" (2006)
  - Lundkvist, "secp256k1 Twist Attacks" (2020)
    https://github.com/mathengem/twist
  - Biehl, Meyer, Muller, "Differential Fault Attacks on Elliptic
    Curve Cryptosystems" (CRYPTO 2000)
"""

import csv
import math
import os
import secrets
import sys
import time

sys.path.insert(0, "src")


# ============================================================================
# Utility functions
# ============================================================================

def jacobi_symbol(a, n):
    """Compute the Jacobi symbol (a/n) for odd n > 0."""
    if n <= 0 or n % 2 == 0:
        raise ValueError("n must be odd and positive")
    a = a % n
    result = 1
    while a != 0:
        while a % 2 == 0:
            a //= 2
            if n % 8 in (3, 5):
                result = -result
        a, n = n, a
        if a % 4 == 3 and n % 4 == 3:
            result = -result
        a = a % n
    return result if n == 1 else 0


def is_quadratic_residue(a, p):
    """Check if a is a quadratic residue mod p (p prime)."""
    if a % p == 0:
        return True
    return pow(a, (p - 1) // 2, p) == 1


def modular_sqrt(a, p):
    """Compute square root of a mod p using Tonelli-Shanks."""
    if a % p == 0:
        return 0
    if not is_quadratic_residue(a, p):
        return None
    if p % 4 == 3:
        return pow(a, (p + 1) // 4, p)
    # Tonelli-Shanks
    q = p - 1
    s = 0
    while q % 2 == 0:
        q //= 2
        s += 1
    z = 2
    while is_quadratic_residue(z, p):
        z += 1
    m = s
    c = pow(z, q, p)
    t = pow(a, q, p)
    r = pow(a, (q + 1) // 2, p)
    while True:
        if t == 1:
            return r
        i = 1
        temp = (t * t) % p
        while temp != 1:
            temp = (temp * temp) % p
            i += 1
        b = pow(c, 1 << (m - i - 1), p)
        m = i
        c = (b * b) % p
        t = (t * c) % p
        r = (r * b) % p


def factorize_small(n, limit=10**6):
    """Factor n by trial division up to limit.

    Returns (factors_dict, cofactor) where cofactor is the
    unfactored remainder (1 if fully factored).
    """
    if n <= 0:
        return {}, n
    factors = {}
    d = 2
    while d * d <= n and d <= limit:
        while n % d == 0:
            factors[d] = factors.get(d, 0) + 1
            n //= d
        d += 1 if d == 2 else 2
    return factors, n


def factor_product_str(factors, cofactor=1):
    """Format a factorization as a readable string."""
    parts = []
    for p in sorted(factors):
        e = factors[p]
        if e == 1:
            parts.append(str(p))
        else:
            parts.append(f"{p}^{e}")
    if cofactor > 1:
        bits = cofactor.bit_length()
        if bits > 80:
            parts.append(f"p{bits}")
        else:
            parts.append(str(cofactor))
    return " * ".join(parts) if parts else "1"


def count_curve_points(p, a, b):
    """Count points on y^2 = x^3 + ax + b over F_p by enumeration."""
    count = 1  # point at infinity
    for x in range(p):
        rhs = (x * x * x + a * x + b) % p
        if rhs == 0:
            count += 1
        elif is_quadratic_residue(rhs, p):
            count += 2
    return count


class SmallEC:
    """Elliptic curve y^2 = x^3 + ax + b over F_p for small primes."""

    def __init__(self, p, a, b, validate=True):
        self.p = p
        self.a = a
        self.b = b
        self.validate = validate
        self._order = None

    @property
    def order(self):
        if self._order is None:
            self._order = count_curve_points(self.p, self.a, self.b)
        return self._order

    def on_curve(self, P):
        if P is None:
            return True
        x, y = P
        return (y * y - x * x * x - self.a * x - self.b) % self.p == 0

    def add(self, P, Q):
        if P is None:
            return Q
        if Q is None:
            return P
        if self.validate:
            if not self.on_curve(P):
                raise ValueError(f"Point {P} not on curve y^2=x^3+{self.a}x+{self.b}")
            if not self.on_curve(Q):
                raise ValueError(f"Point {Q} not on curve y^2=x^3+{self.a}x+{self.b}")
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

    def find_point(self, start_x=1):
        """Find a point on this curve."""
        for x in range(start_x, self.p):
            rhs = (x * x * x + self.a * x + self.b) % self.p
            y = modular_sqrt(rhs, self.p)
            if y is not None:
                return (x, y)
        return None

    def find_point_of_order(self, target_order):
        """Find a point whose order divides target_order.

        Generates random points and multiplies by cofactor = order / target_order.
        """
        if self.order % target_order != 0:
            return None
        cofactor = self.order // target_order
        for x in range(1, self.p):
            rhs = (x * x * x + self.a * x + self.b) % self.p
            y = modular_sqrt(rhs, self.p)
            if y is not None:
                P = (x, y)
                Q = self.multiply(P, cofactor)
                if Q is not None and self.multiply(Q, target_order) is None:
                    return Q
        return None


def find_twist_curve(p, a, b):
    """Find the quadratic twist of y^2 = x^3 + ax + b.

    The twist is y^2 = x^3 + a*d^2*x + b*d^3 where d is a non-square mod p.
    Returns (a', b', d) for the twist.
    """
    # Find the smallest non-residue d
    d = 2
    while is_quadratic_residue(d, p):
        d += 1
    d2 = (d * d) % p
    d3 = (d2 * d) % p
    a_twist = (a * d2) % p
    b_twist = (b * d3) % p
    return a_twist, b_twist, d


def find_twist_point(p, a, b, start_x=0):
    """Find an x-coordinate that lands on the twist (not on the curve).

    Returns x such that x^3 + ax + b is NOT a QR mod p.
    The 'point' (x, ?) is on the twist, not on E.
    """
    for x in range(start_x, p):
        rhs = (x * x * x + a * x + b) % p
        if rhs != 0 and not is_quadratic_residue(rhs, p):
            return x
    return None


# ============================================================================
# PART 1: Quadratic Twist Basics on Small Curves
# ============================================================================

def part1_small_curve_twists():
    print()
    print("=" * 78)
    print("  PART 1: Quadratic Twist Basics on Small Curves")
    print("  E: y^2 = x^3 + 7 over F_p (secp256k1 equation, small primes)")
    print("=" * 78)

    test_primes = [23, 47, 67, 101, 251, 503]

    print()
    print(f"  {'p':>5}  {'|E|':>6}  {'|E_twist|':>10}  {'|E|+|E_t|':>10}  "
          f"{'2p+2':>10}  {'Twist Factorization':<30}  {'Security'}")
    print(f"  {'---':>5}  {'---':>6}  {'---':>10}  {'---':>10}  "
          f"{'---':>10}  {'---':<30}  {'---'}")

    results = []

    for p in test_primes:
        # Count points on E: y^2 = x^3 + 7
        E_order = count_curve_points(p, 0, 7)
        # Twist order from Hasse: |E| + |E'| = 2p + 2
        twist_order = 2 * p + 2 - E_order
        hasse_sum = E_order + twist_order

        # Factor the twist order
        factors, cofactor = factorize_small(twist_order)
        factor_str = factor_product_str(factors, cofactor)

        # Classify security
        if cofactor == 1:
            # Fully factored
            largest_prime = max(factors.keys()) if factors else 1
        else:
            largest_prime = cofactor  # unfactored part is presumably prime
        if largest_prime > p // 2:
            security = "SECURE"
        elif largest_prime > 20:
            security = "moderate"
        else:
            security = "WEAK"

        print(f"  {p:>5}  {E_order:>6}  {twist_order:>10}  {hasse_sum:>10}  "
              f"{2*p+2:>10}  {factor_str:<30}  {security}")

        results.append({
            "p": p,
            "curve_order": E_order,
            "twist_order": twist_order,
            "twist_factors": factor_str,
            "largest_factor": largest_prime,
            "security": security,
        })

    print()
    print("  Verification: |E| + |E'| = 2p + 2 for all primes (Hasse's theorem).")
    print()
    print("  'SECURE' = largest prime factor > p/2 (Pollard rho on twist is hard)")
    print("  'WEAK'   = twist order is smooth (all small factors, easy DLP on twist)")
    print("  'moderate' = mixed -- some small factors exist but largest is nontrivial")

    return results


# ============================================================================
# PART 2: secp256k1 Twist Analysis
# ============================================================================

def part2_secp256k1_twist():
    print()
    print()
    print("=" * 78)
    print("  PART 2: secp256k1 Quadratic Twist Analysis")
    print("=" * 78)

    p = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
    n = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141

    twist_order = 2 * p + 2 - n

    print(f"""
  secp256k1 parameters:
    p = {p}
    n = {n}
    h = 1 (cofactor -- group order equals subgroup order, n is prime)

  Quadratic twist order (from |E| + |E'| = 2p + 2):
    n' = 2p + 2 - n
    n' = {twist_order}
    n' bits = {twist_order.bit_length()}
""")

    # Factor by trial division
    factors, cofactor = factorize_small(twist_order, limit=10**6)

    print(f"  Trial division up to 10^6:")
    print(f"    Small prime factors found: ", end="")
    parts = []
    for prime in sorted(factors):
        exp = factors[prime]
        if exp == 1:
            parts.append(str(prime))
        else:
            parts.append(f"{prime}^{exp}")
    print(" * ".join(parts))

    small_product = 1
    for prime, exp in factors.items():
        small_product *= prime ** exp

    print(f"    Product of small factors: {small_product}")
    print(f"    Remaining cofactor: {cofactor}")
    print(f"    Remaining cofactor bits: {cofactor.bit_length()}")

    # Verify against known factorization from SafeCurves
    known_small = 3**2 * 13**2 * 3319 * 22639
    known_large = 1013176677300131846900870239606035638738100997248092069256697437031

    print(f"""
  Known factorization (SafeCurves / Bernstein-Lange):
    n' = 3^2 * 13^2 * 3319 * 22639 * p220
    where p220 = {known_large}
    p220 is a 220-bit prime

  Verification:
    3^2 * 13^2 * 3319 * 22639 = {known_small}
    {known_small} * p220 = {known_small * known_large}
    Match: {known_small * known_large == twist_order}
""")

    # Security analysis
    rho_bits = known_large.bit_length() / 2.0
    print(f"  Twist security assessment:")
    print(f"    Largest prime factor: {known_large.bit_length()} bits")
    print(f"    Pollard rho cost on twist: ~2^{rho_bits:.1f} operations")
    print(f"    SafeCurves threshold: 2^100")
    print(f"    Status: {'PASS' if rho_bits >= 100 else 'FAIL'} "
          f"(2^{rho_bits:.1f} {'>' if rho_bits >= 100 else '<'} 2^100)")

    print(f"""
  What this means:
    - The twist has small factors: 3^2, 13^2, 3319, 22639
    - An attacker sending a twist point could learn k mod 9, k mod 169,
      k mod 3319, or k mod 22639 (total ~37 bits of information)
    - But the 220-bit prime factor means the FULL twist DLP is infeasible
    - To recover the key completely, the attacker still needs to solve a
      220-bit DLP on the twist, which costs ~2^110 operations
    - This is above the SafeCurves 2^100 threshold: twist is adequately secure
    - HOWEVER: the small factors DO leak partial information without validation
""")

    return {
        "twist_order": twist_order,
        "small_factors": factors,
        "cofactor": cofactor,
        "known_large_prime": known_large,
        "rho_bits": rho_bits,
    }


# ============================================================================
# PART 3: Twist Attack Simulation on Small Curves
# ============================================================================

def part3_twist_attack_simulation():
    print()
    print("=" * 78)
    print("  PART 3: Twist Attack Simulation on Small Curves")
    print("=" * 78)

    print("""
  Attack scenario:
    1. Attacker finds x where x^3 + 7 is NOT a quadratic residue mod p
       (this x-coordinate lies on the TWIST, not on the real curve)
    2. If the implementation does x-only scalar multiplication without
       checking that the point is on E, it computes on the twist instead
    3. The result leaks k mod (small factor of twist order)
    4. CRT combines partial leaks to recover key bits
""")

    # Pick a prime where the twist has interesting factors
    demo_primes = [101, 251, 503]

    for p in demo_primes:
        print(f"  --- Field F_{p} ---")

        E = SmallEC(p, 0, 7, validate=False)
        E_order = E.order
        twist_order = 2 * p + 2 - E_order
        factors_twist, cof_twist = factorize_small(twist_order)

        # Find twist parameters
        a_t, b_t, d = find_twist_curve(p, 0, 7)
        E_twist = SmallEC(p, a_t, b_t, validate=False)
        E_twist_order = E_twist.order

        print(f"    Curve E: y^2 = x^3 + 7, |E| = {E_order}")
        print(f"    Twist E': y^2 = x^3 + {a_t}x + {b_t} (d={d}), |E'| = {E_twist_order}")
        print(f"    Hasse check: {E_order} + {E_twist_order} = {E_order + E_twist_order} "
              f"= 2*{p}+2 = {2*p+2}: {'OK' if E_order + E_twist_order == 2*p+2 else 'FAIL'}")
        print(f"    Twist factorization: {factor_product_str(factors_twist, cof_twist)}")

        # Find small prime factors of the twist
        small_primes = [q for q in factors_twist if q <= 100]

        if not small_primes:
            print(f"    No small prime factors in twist -- twist is secure for this prime.")
            print()
            continue

        # Simulate the attack
        victim_key = secrets.randbelow(E_order - 1) + 1
        print(f"    Victim's secret key: k = {victim_key}")
        print()

        residues = []
        moduli = []

        for q in small_primes:
            # Find a point of order q on the twist
            P_twist = E_twist.find_point_of_order(q)
            if P_twist is None:
                print(f"    Could not find order-{q} point on twist, skipping")
                continue

            # Verify order
            check = E_twist.multiply(P_twist, q)
            if check is not None:
                continue

            # The attacker "sends" a twist point
            # The victim computes k * P_twist on the twist (no validation!)
            result = E_twist.multiply(P_twist, victim_key)

            # The attacker brute-forces k mod q
            k_mod_q = None
            for d_try in range(q):
                if E_twist.multiply(P_twist, d_try) == result:
                    k_mod_q = d_try
                    break

            actual_mod = victim_key % q
            print(f"    Twist point of order {q}: {P_twist}")
            print(f"      Victim computes: k * P_twist = {result}")
            print(f"      Attacker recovers: k mod {q} = {k_mod_q}")
            print(f"      Actual k mod {q} = {actual_mod}")
            print(f"      Correct: {k_mod_q == actual_mod}")

            if k_mod_q is not None:
                residues.append(k_mod_q)
                moduli.append(q)

        # CRT recovery
        if len(residues) >= 2:
            M = 1
            for m in moduli:
                M *= m
            k_partial = 0
            for r, m in zip(residues, moduli):
                Mi = M // m
                yi = pow(Mi, -1, m)
                k_partial = (k_partial + r * Mi * yi) % M

            print(f"\n    CRT combination:")
            print(f"      Moduli: {moduli}, product M = {M}")
            print(f"      Recovered: k mod {M} = {k_partial}")
            print(f"      Actual k mod {M} = {victim_key % M}")
            print(f"      Correct: {k_partial == victim_key % M}")
            bits_leaked = math.log2(M) if M > 1 else 0
            bits_total = math.log2(E_order) if E_order > 1 else 0
            print(f"      Bits leaked: {bits_leaked:.1f} out of {bits_total:.1f} "
                  f"({100*bits_leaked/bits_total:.1f}%)")

        print()

    # Now compare: curve with secure twist vs insecure twist
    print(f"\n  --- Comparison: Secure vs Insecure Twist ---")
    print()

    # Find a prime where twist of y^2=x^3+7 is smooth vs large prime
    print(f"  Scanning small primes for twist security classification...")
    secure_examples = []
    weak_examples = []

    for p in range(23, 600, 2):
        # Check primality simply
        if any(p % d == 0 for d in range(2, int(p**0.5) + 1)):
            continue
        E_ord = count_curve_points(p, 0, 7)
        tw_ord = 2 * p + 2 - E_ord
        if tw_ord <= 1:
            continue
        facs, cof = factorize_small(tw_ord)
        largest = cof if cof > 1 else (max(facs) if facs else 1)
        is_smooth = (cof == 1 and largest <= 50)

        if is_smooth and tw_ord > 10 and len(weak_examples) < 3:
            weak_examples.append((p, E_ord, tw_ord, facs, cof))
        if not is_smooth and largest > p // 3 and len(secure_examples) < 3:
            secure_examples.append((p, E_ord, tw_ord, facs, cof))

    print(f"\n  Curves with WEAK twists (smooth twist order):")
    for p, eo, to, facs, cof in weak_examples:
        print(f"    p={p}: |E|={eo}, |E'|={to} = {factor_product_str(facs, cof)}")
        print(f"      -> All factors tiny, twist DLP is trivial")
        print(f"      -> Full key leakage possible through twist points")

    print(f"\n  Curves with SECURE twists (large prime in twist order):")
    for p, eo, to, facs, cof in secure_examples:
        print(f"    p={p}: |E|={eo}, |E'|={to} = {factor_product_str(facs, cof)}")
        print(f"      -> Large prime factor blocks full key recovery via twist")


# ============================================================================
# PART 4: X-only Protocols and Twist Safety
# ============================================================================

def part4_xonly_protocols():
    print()
    print()
    print("=" * 78)
    print("  PART 4: X-only Protocols and Twist Safety")
    print("=" * 78)

    print("""
  In modern protocols, only x-coordinates are often transmitted:

  X25519 (Curve25519 Diffie-Hellman):
  -----------------------------------
    - Montgomery curve: By^2 = x^3 + Ax^2 + x
    - The Montgomery ladder computes x-coordinate of scalar multiple
      WITHOUT needing the y-coordinate at all
    - But: given just an x-coordinate, there is no way to tell if the
      point is on the curve or on the twist
    - The ladder WORKS on both -- it computes the correct x-coordinate
      of k*P regardless of which curve P is actually on
    - Therefore: security requires BOTH curve AND twist to have large
      prime-order subgroups

  Curve25519 design (Bernstein 2006):
  -----------------------------------
    - Curve order:  8 * large_prime  (cofactor 8)
    - Twist order:  4 * large_prime  (cofactor 4)
    - BOTH have huge prime factors (>2^250)
    - This was a DELIBERATE design choice
    - The small cofactors (8 and 4) are handled by clamping the scalar:
      set the low 3 bits to 0 (multiply by 8 effectively)
    - Result: even unchecked twist points cannot leak key information

  secp256k1 (Bitcoin, Ethereum):
  -----------------------------------
    - Weierstrass form: y^2 = x^3 + 7
    - Standard EC addition formulas REQUIRE both coordinates
    - Point validation checks y^2 = x^3 + 7 before any computation
    - Compressed points: store x and a parity bit, solve for y
      -> decompression inherently validates (if no valid y exists, reject)
    - Twist order = 3^2 * 13^2 * 3319 * 22639 * p220
    - The small factors (up to 22639) could leak ~37 bits without validation
    - libsecp256k1 always validates -> twist attack is blocked

  Taproot (BIP 340/341, BIP 327 MuSig2):
  -----------------------------------
    - Uses x-only public keys (32 bytes instead of 33)
    - Implicit y parity: always choose the even y-coordinate
    - During verification: reconstruct full point from x, check on curve
    - If x^3 + 7 has no square root mod p, the public key is INVALID
    - This means every Taproot x-only pubkey is validated against the twist
    - No twist attack is possible in compliant implementations

  Ed448 (EdDSA with Curve448):
  -----------------------------------
    - Edwards form: x^2 + y^2 = 1 + d*x^2*y^2
    - Curve order:  4 * large_prime  (cofactor 4)
    - Twist order:  4 * large_prime  (cofactor 4)
    - Same twist-secure design philosophy as Curve25519
    - Points validated during decoding (reject non-curve points)
""")

    # Demonstrate on small curves: x-only ambiguity
    p = 101
    E = SmallEC(p, 0, 7, validate=False)
    a_t, b_t, d = find_twist_curve(p, 0, 7)
    E_tw = SmallEC(p, a_t, b_t, validate=False)

    print(f"  Demonstration of x-coordinate ambiguity (p = {p}):")
    print(f"    Curve E: y^2 = x^3 + 7")
    print(f"    Twist E': y^2 = x^3 + {a_t}x + {b_t}")
    print()

    n_on_curve = 0
    n_on_twist = 0
    n_both = 0  # x = 0 might be on both

    for x in range(p):
        rhs = (x * x * x + 7) % p
        if rhs == 0:
            n_on_curve += 1
            n_on_twist += 1
            n_both += 1
        elif is_quadratic_residue(rhs, p):
            n_on_curve += 1
        else:
            n_on_twist += 1

    print(f"    x-coordinates on E: {n_on_curve} / {p}")
    print(f"    x-coordinates on E': {n_on_twist} / {p}")
    print(f"    On both (rhs=0): {n_both} / {p}")
    print(f"    Total: {n_on_curve + n_on_twist - n_both} = {p} (every x lands somewhere)")
    print()
    print(f"  Roughly half of all x-coordinates land on the twist.")
    print(f"  Without validation, an attacker can freely choose twist points.")


# ============================================================================
# PART 5: Comparison Table of Standard Curves
# ============================================================================

def part5_comparison_table():
    print()
    print()
    print("=" * 78)
    print("  PART 5: Twist Security Comparison of Standard Curves")
    print("=" * 78)

    # All values from published standards and SafeCurves
    curves = []

    # --- secp256k1 ---
    secp256k1_p = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
    secp256k1_n = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
    secp256k1_tw = 2 * secp256k1_p + 2 - secp256k1_n
    secp256k1_tw_small = 3**2 * 13**2 * 3319 * 22639
    secp256k1_tw_large = 1013176677300131846900870239606035638738100997248092069256697437031

    curves.append({
        "name": "secp256k1",
        "field_bits": secp256k1_p.bit_length(),
        "p": secp256k1_p,
        "curve_order": secp256k1_n,
        "curve_cofactor": 1,
        "twist_order": secp256k1_tw,
        "twist_smallest_factor": 3,
        "twist_cofactor_smooth": secp256k1_tw_small,
        "twist_largest_prime_bits": secp256k1_tw_large.bit_length(),
        "twist_rho_bits": secp256k1_tw_large.bit_length() / 2.0,
        "twist_secure": True,
        "form": "Weierstrass",
        "notes": "Twist has small factors 3^2*13^2*3319*22639 but 220-bit prime; "
                 "validation required; libsecp256k1 validates",
    })

    # --- secp256r1 (P-256) ---
    p256_p = 0xffffffff00000001000000000000000000000000ffffffffffffffffffffffff
    p256_n = 0xffffffff00000000ffffffffffffffffbce6faada7179e84f3b9cac2fc632551
    p256_tw = 2 * p256_p + 2 - p256_n
    p256_tw_small = 3 * 5 * 13 * 179
    p256_tw_large = 3317349640749355357762425066592395746459685764401801118712075735758936647

    curves.append({
        "name": "secp256r1 (P-256)",
        "field_bits": p256_p.bit_length(),
        "p": p256_p,
        "curve_order": p256_n,
        "curve_cofactor": 1,
        "twist_order": p256_tw,
        "twist_smallest_factor": 3,
        "twist_cofactor_smooth": p256_tw_small,
        "twist_largest_prime_bits": p256_tw_large.bit_length(),
        "twist_rho_bits": p256_tw_large.bit_length() / 2.0,
        "twist_secure": True,
        "form": "Weierstrass",
        "notes": "Twist cofactor 3*5*13*179=34905; 241-bit prime; NIST curve; "
                 "validation required",
    })

    # --- Curve25519 ---
    c25519_p = 2**255 - 19
    c25519_ell = 7237005577332262213973186563042994240857116359379907606001950938285454250989
    c25519_order = 8 * c25519_ell
    c25519_tw = 2 * c25519_p + 2 - c25519_order
    c25519_tw_large = c25519_tw // 4  # twist cofactor is 4

    curves.append({
        "name": "Curve25519",
        "field_bits": c25519_p.bit_length(),
        "p": c25519_p,
        "curve_order": c25519_order,
        "curve_cofactor": 8,
        "twist_order": c25519_tw,
        "twist_smallest_factor": 2,
        "twist_cofactor_smooth": 4,
        "twist_largest_prime_bits": c25519_tw_large.bit_length(),
        "twist_rho_bits": c25519_tw_large.bit_length() / 2.0,
        "twist_secure": True,
        "form": "Montgomery",
        "notes": "Designed twist-secure; twist cofactor=4; 253-bit prime; "
                 "x-only Montgomery ladder safe on both curve and twist",
    })

    # --- Ed448 (Curve448-Goldilocks) ---
    ed448_p = 2**448 - 2**224 - 1
    ed448_ell = 2**446 - 13818066809895115352007386748515426880336692474882178609894547503885
    ed448_order = 4 * ed448_ell
    ed448_tw = 2 * ed448_p + 2 - ed448_order
    ed448_tw_large = ed448_tw // 4

    curves.append({
        "name": "Ed448 (Curve448)",
        "field_bits": ed448_p.bit_length(),
        "p": ed448_p,
        "curve_order": ed448_order,
        "curve_cofactor": 4,
        "twist_order": ed448_tw,
        "twist_smallest_factor": 2,
        "twist_cofactor_smooth": 4,
        "twist_largest_prime_bits": ed448_tw_large.bit_length(),
        "twist_rho_bits": ed448_tw_large.bit_length() / 2.0,
        "twist_secure": True,
        "form": "Edwards",
        "notes": "Designed twist-secure; twist cofactor=4; 447-bit prime; "
                 "same design philosophy as Curve25519",
    })

    # --- secp384r1 (P-384) for reference ---
    p384_p = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFFFF0000000000000000FFFFFFFF
    p384_n = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFC7634D81F4372DDF581A0DB248B0A77AECEC196ACCC52973
    p384_tw = 2 * p384_p + 2 - p384_n
    # Trial factor
    p384_factors, p384_cof = factorize_small(p384_tw, limit=10**6)
    p384_small_product = 1
    for prime, exp in p384_factors.items():
        p384_small_product *= prime ** exp
    p384_smallest = min(p384_factors.keys()) if p384_factors else p384_cof

    curves.append({
        "name": "secp384r1 (P-384)",
        "field_bits": p384_p.bit_length(),
        "p": p384_p,
        "curve_order": p384_n,
        "curve_cofactor": 1,
        "twist_order": p384_tw,
        "twist_smallest_factor": p384_smallest,
        "twist_cofactor_smooth": p384_small_product,
        "twist_largest_prime_bits": p384_cof.bit_length() if p384_cof > 1 else 0,
        "twist_rho_bits": (p384_cof.bit_length() if p384_cof > 1 else 0) / 2.0,
        "twist_secure": (p384_cof.bit_length() if p384_cof > 1 else 0) / 2.0 >= 100,
        "form": "Weierstrass",
        "notes": f"Twist small factors: {factor_product_str(p384_factors)}; "
                 f"cofactor {p384_cof.bit_length()}-bit",
    })

    # Print comparison table
    print()
    print(f"  {'Curve':<20} {'Bits':>4}  {'Form':<12} {'h':>2}  "
          f"{'Twist h_smooth':<18} {'Twist rho':>10}  {'Safe?':>5}  "
          f"{'Leak (bits)':<12}")
    print(f"  {'-'*20} {'----':>4}  {'-'*12} {'--':>2}  "
          f"{'-'*18} {'-'*10}  {'-----':>5}  "
          f"{'-'*12}")

    for c in curves:
        leak_bits = math.log2(c["twist_cofactor_smooth"]) if c["twist_cofactor_smooth"] > 1 else 0
        print(f"  {c['name']:<20} {c['field_bits']:>4}  {c['form']:<12} "
              f"{c['curve_cofactor']:>2}  "
              f"{c['twist_cofactor_smooth']:<18}  "
              f"2^{c['twist_rho_bits']:.1f}     "
              f"{'YES' if c['twist_secure'] else 'NO':>5}  "
              f"~{leak_bits:.0f} bits")

    print()
    print("  Column explanation:")
    print("    Bits:          Field size in bits")
    print("    Form:          Curve representation (Weierstrass/Montgomery/Edwards)")
    print("    h:             Curve cofactor (1 = prime order)")
    print("    Twist h_smooth: Product of small factors in the twist order")
    print("    Twist rho:     Pollard rho cost on the twist's largest prime subgroup")
    print("    Safe?:         Twist rho >= 2^100 (SafeCurves criterion)")
    print("    Leak (bits):   Information leaked per twist query without validation")

    print(f"""
  Key observations:
    1. Curve25519 and Ed448 were DESIGNED to be twist-secure:
       - Twist cofactors are just 4 (two bits), not exploitable
       - This is WHY they use Montgomery/Edwards forms with x-only protocols

    2. secp256k1 and P-256 have twist cofactors with multiple small factors:
       - secp256k1: 3^2 * 13^2 * 3319 * 22639 = {secp256k1_tw_small} (~37 bits leakable)
       - P-256: 3 * 5 * 13 * 179 = {p256_tw_small} (~15 bits leakable)
       - BUT their largest twist prime factors are huge (220+ bits)
       - So full key recovery via twist alone is infeasible

    3. The twist security difference reflects DESIGN PHILOSOPHY:
       - Curve25519/Ed448: designed for x-only protocols, must be twist-safe
       - secp256k1/P-256: Weierstrass form with mandatory point validation,
         twist security is nice-to-have but not strictly required
       - All four curves pass the SafeCurves twist security criterion
""")

    return curves


# ============================================================================
# CSV Output
# ============================================================================

def write_csv(curves_data):
    """Write the comparison data to CSV."""
    desktop = os.path.expanduser("~/Desktop")
    csv_path = os.path.join(desktop, "twist_security.csv")

    rows = []
    for c in curves_data:
        rows.append({
            "curve_name": c["name"],
            "field_size": c["field_bits"],
            "curve_order": str(c["curve_order"]),
            "twist_order": str(c["twist_order"]),
            "twist_smallest_factor": c["twist_smallest_factor"],
            "twist_largest_factor_bits": c["twist_largest_prime_bits"],
            "twist_secure": "yes" if c["twist_secure"] else "no",
            "notes": c["notes"],
        })

    with open(csv_path, "w", newline="") as f:
        fieldnames = [
            "curve_name", "field_size", "curve_order", "twist_order",
            "twist_smallest_factor", "twist_largest_factor_bits",
            "twist_secure", "notes",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    return csv_path


# ============================================================================
# SUMMARY
# ============================================================================

def print_summary():
    print()
    print("=" * 78)
    print("  SUMMARY: Why Twist Security Matters for Protocol Design")
    print("=" * 78)
    print("""
  The quadratic twist of an elliptic curve is not an abstract curiosity --
  it is the UNAVOIDABLE companion curve that shares the same x-coordinates.
  Every x in F_p either has y^2 = x^3 + ax + b a quadratic residue (point
  on E) or a non-residue (point on the twist E'). There is no third option.

  WHY TWIST SECURITY MATTERS:

  1. Invalid input handling:
     If a protocol receives a bare x-coordinate and computes scalar
     multiplication without checking which curve the point is on, the
     computation silently happens on the twist. If the twist has small
     subgroups, partial key bits leak.

  2. X-only protocols (the modern trend):
     X25519, Taproot (BIP 340), and other modern protocols transmit only
     x-coordinates for bandwidth efficiency. The Montgomery ladder used
     in X25519 works identically on both curve and twist. This makes
     twist security a HARD REQUIREMENT, not optional.

  3. Side-channel resistance:
     Montgomery ladder and similar constant-time algorithms don't branch
     on whether the point is on E or E'. They produce a valid result
     either way. The security assumption is that both results are hard
     to invert.

  HOW secp256k1 HANDLES IT:

  - secp256k1 uses Weierstrass form, requiring full (x, y) points
  - libsecp256k1 validates every point: checks y^2 = x^3 + 7 mod p
  - Compressed points (33 bytes): decompression rejects twist points
  - Taproot x-only (32 bytes): verification reconstructs y, rejects if impossible
  - Result: twist attacks are completely blocked in Bitcoin

  THE DESIGN TRADEOFF:

  +------------------+-------------------+----------------------------+
  | Property         | Curve25519/Ed448  | secp256k1/P-256           |
  +------------------+-------------------+----------------------------+
  | Twist cofactor   | 4 (harmless)      | Large smooth part          |
  | Validation need  | Not required      | Required (must check y)    |
  | X-only safe      | Yes (by design)   | Yes (with validation)      |
  | Attack surface   | Minimal           | Implementation-dependent   |
  +------------------+-------------------+----------------------------+

  Curves like Curve25519 achieve "misuse resistance" -- even a buggy
  implementation that skips validation leaks at most 2 bits (the cofactor).
  secp256k1 achieves security through MANDATORY validation, which is
  effective but creates a larger attack surface for implementation errors.

  BOTTOM LINE:
  - All four major curves (secp256k1, P-256, Curve25519, Ed448) have
    adequate twist security (rho cost > 2^100)
  - Curve25519 and Ed448 have EXCEPTIONAL twist security by design
  - secp256k1 and P-256 require correct implementation to be safe
  - Bitcoin's libsecp256k1 implements all necessary checks
  - The twist is not a vulnerability in secp256k1 itself, but a reason
    why point validation is non-negotiable in Weierstrass-form protocols
""")


# ============================================================================
# Main
# ============================================================================

def main():
    print()
    print("*" * 78)
    print("*" + " " * 76 + "*")
    print("*  QUADRATIC TWIST SECURITY ANALYSIS" + " " * 40 + "*")
    print("*  secp256k1, P-256, Curve25519, Ed448" + " " * 38 + "*")
    print("*" + " " * 76 + "*")
    print("*" * 78)

    t_start = time.time()

    # Part 1: Small curve basics
    part1_small_curve_twists()

    # Part 2: secp256k1 deep dive
    part2_secp256k1_twist()

    # Part 3: Attack simulation
    part3_twist_attack_simulation()

    # Part 4: X-only protocols
    part4_xonly_protocols()

    # Part 5: Comparison table
    curves_data = part5_comparison_table()

    # Summary
    print_summary()

    # CSV output
    csv_path = write_csv(curves_data)

    elapsed = time.time() - t_start

    print("=" * 78)
    print(f"  CSV written to: {csv_path}")
    print(f"  Total runtime: {elapsed:.2f}s")
    print("=" * 78)


if __name__ == "__main__":
    main()
