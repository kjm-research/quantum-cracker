"""Multi-Target Batch DLP Attack.

When attacking T targets simultaneously, can we amortize work?

Key insight (Kuhn & Struik 2001, Hitchcock et al. 2006):
- Single target Pollard rho: O(sqrt(N)) per target, O(T * sqrt(N)) total
- Multi-target with distinguished points: O(sqrt(N*T)) total
- Speedup: sqrt(T) over independent attacks

For Bitcoin: if you want to crack ANY ONE of T known public keys,
the cost drops by sqrt(T). With T = 2^40 (~1 trillion) exposed keys,
the cost drops from 2^128 to 2^108. Still infeasible, but interesting.

This script implements multi-target Pollard rho with distinguished
points on small EC curves.
"""

import csv
import math
import os
import secrets
import sys
import time

import numpy as np

sys.path.insert(0, "src")


class SmallEC:
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
            self._find_gen()
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
        self._pts = pts

    def _find_gen(self):
        if self._order is None:
            self._enumerate()
        for pt in self._pts[1:]:
            if self.multiply(pt, self.order) is None:
                is_gen = True
                for d in range(2, int(self.order ** 0.5) + 1):
                    if self.order % d == 0:
                        if self.multiply(pt, self.order // d) is None:
                            is_gen = False
                            break
                if is_gen:
                    self._gen = pt
                    return pt
        self._gen = self._pts[1]
        return self._gen

    def add(self, P, Q):
        if P is None: return Q
        if Q is None: return P
        p = self.p
        x1, y1 = P
        x2, y2 = Q
        if x1 == x2 and y1 == (p - y2) % p: return None
        if P == Q:
            if y1 == 0: return None
            lam = (3 * x1 * x1 + self.a) * pow(2 * y1, p - 2, p) % p
        else:
            if x1 == x2: return None
            lam = (y2 - y1) * pow((x2 - x1) % p, p - 2, p) % p
        x3 = (lam * lam - x1 - x2) % p
        y3 = (lam * (x1 - x3) - y1) % p
        return (x3, y3)

    def neg(self, P):
        if P is None: return None
        return (P[0], (self.p - P[1]) % self.p)

    def multiply(self, P, k):
        if k < 0:
            P = self.neg(P)
            k = -k
        if k == 0 or P is None: return None
        result = None
        addend = P
        while k:
            if k & 1: result = self.add(result, addend)
            addend = self.add(addend, addend)
            k >>= 1
        return result


def _prime_factors(n):
    """Return the set of distinct prime factors of n."""
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


def _divisors(n):
    """Return all positive divisors of n."""
    divs = set()
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divs.add(i)
            divs.add(n // i)
    return divs


def _subgroup_order(ec, G):
    """Compute the order of point G in the group."""
    n = ec.order
    divs = sorted(_divisors(n))
    for d in divs:
        if d > 0 and ec.multiply(G, d) is None:
            return d
    return n


def is_distinguished(point, threshold_bits):
    """A point is 'distinguished' if its x-coordinate has
    `threshold_bits` trailing zeros."""
    if point is None:
        return True
    return (point[0] & ((1 << threshold_bits) - 1)) == 0


def single_target_rho(ec, G, Q, max_ops, group_order=None):
    """Pollard's rho for single-target ECDLP with Floyd's cycle detection.

    Uses the correct subgroup order (not ec.order - 1) and handles
    non-prime group orders via GCD-based collision resolution.
    Adapted from pollard_rho_kangaroo_fixed.py.
    """
    if group_order is None:
        group_order = _subgroup_order(ec, G)
    n = group_order
    if n <= 1:
        return 0, 0
    if Q is None:
        return 0, 0
    if Q == G:
        return 1, 0

    total_ops = 0
    max_restarts = 5
    max_iter_per_restart = max(max_ops // max_restarts, 4 * int(math.isqrt(n)) + 100)

    def partition(P):
        if P is None:
            return 0
        return P[0] % 3

    def step(R, a, b):
        s = partition(R)
        if s == 0:
            R = ec.add(R, Q)
            b = (b + 1) % n
        elif s == 1:
            R = ec.add(R, R)
            a = (a * 2) % n
            b = (b * 2) % n
        else:
            R = ec.add(R, G)
            a = (a + 1) % n
        return R, a, b

    for restart in range(max_restarts):
        a0 = secrets.randbelow(n) if n > 1 else 0
        b0 = secrets.randbelow(n) if n > 1 else 0
        if a0 == 0 and b0 == 0:
            a0 = 1
        R0 = ec.add(ec.multiply(G, a0), ec.multiply(Q, b0))
        total_ops += 2

        a_t, b_t, R_t = a0, b0, R0
        a_h, b_h, R_h = a0, b0, R0

        for iteration in range(max_iter_per_restart):
            R_t, a_t, b_t = step(R_t, a_t, b_t)
            total_ops += 1
            R_h, a_h, b_h = step(R_h, a_h, b_h)
            R_h, a_h, b_h = step(R_h, a_h, b_h)
            total_ops += 2

            if R_t == R_h:
                # Collision: a_t*G + b_t*Q = a_h*G + b_h*Q
                # => k = (a_t - a_h) / (b_h - b_t) mod n
                db = (b_h - b_t) % n
                da = (a_t - a_h) % n

                if db == 0:
                    break  # useless collision, restart

                g = math.gcd(db, n)
                if g == 1:
                    try:
                        db_inv = pow(db, -1, n)
                        k = (da * db_inv) % n
                        if ec.multiply(G, k) == Q:
                            return k, total_ops
                    except (ValueError, ZeroDivisionError):
                        break
                else:
                    # gcd > 1: try all g possible solutions
                    if da % g != 0:
                        break
                    da_red = da // g
                    db_red = db // g
                    n_red = n // g
                    try:
                        db_red_inv = pow(db_red, -1, n_red)
                    except (ValueError, ZeroDivisionError):
                        break
                    base_k = (da_red * db_red_inv) % n_red
                    for j in range(g):
                        k_candidate = (base_k + j * n_red) % n
                        if ec.multiply(G, k_candidate) == Q:
                            return k_candidate, total_ops
                    break

    return None, total_ops


def _try_extract_dlog(ec, G, Q, da, db, n):
    """Try to extract discrete log k from collision da*G = db*Q mod n.

    Handles both prime and composite group orders via GCD resolution.
    Returns k if found, else None.
    """
    if db == 0:
        return None

    g = math.gcd(db, n)
    if g == 1:
        try:
            db_inv = pow(db, -1, n)
            k = (da * db_inv) % n
            if ec.multiply(G, k) == Q:
                return k
        except (ValueError, ZeroDivisionError):
            return None
    else:
        if da % g != 0:
            return None
        da_red = da // g
        db_red = db // g
        n_red = n // g
        try:
            db_red_inv = pow(db_red, -1, n_red)
        except (ValueError, ZeroDivisionError):
            return None
        base_k = (da_red * db_red_inv) % n_red
        for j in range(g):
            k_candidate = (base_k + j * n_red) % n
            if ec.multiply(G, k_candidate) == Q:
                return k_candidate
    return None


def multi_target_rho(ec, G, targets, max_ops, group_order=None):
    """Multi-target Pollard rho with distinguished points.

    Walk randomly in the group. At distinguished points, store (point, a, b).
    When two walks for the same target collide at a distinguished point,
    we extract the DLP from the collision.

    Uses the correct subgroup order and GCD-aware collision resolution.

    targets: list of (Q_i, k_i) where Q_i = k_i * G
    """
    if group_order is None:
        group_order = _subgroup_order(ec, G)
    n = group_order
    if n <= 1:
        return {}, 0

    T = len(targets)

    # Distinguished point threshold: ~sqrt(n/T) steps between DPs
    dp_bits = max(1, int(math.log2(max(n, 2)) / 2 - math.log2(max(T, 1)) / 2))

    # Storage for distinguished points, keyed by (target_index, point)
    dp_store = {}  # (target_i, point_key) -> (a, b)

    # Initialize multiple walks per target for better collision probability
    n_walks_per_target = max(2, int(math.sqrt(T)) + 1)
    walks = []
    target_map = {}
    solved = {}

    for i, (Q_i, _) in enumerate(targets):
        target_map[i] = Q_i
        for _ in range(n_walks_per_target):
            a = secrets.randbelow(max(n, 1))
            b = secrets.randbelow(max(n, 1))
            if a == 0 and b == 0:
                a = 1
            R = ec.add(ec.multiply(G, a), ec.multiply(Q_i, b))
            walks.append((R, a, b, i))

    ops = 0

    for iteration in range(max_ops):
        new_walks = []
        for R, a, b, target_i in walks:
            if target_i in solved:
                continue

            Q_i = target_map[target_i]

            # Take one step (3-partition random walk)
            if R is None:
                s = 0
            else:
                s = R[0] % 3

            if s == 0:
                R_new = ec.add(R, Q_i)
                a_new, b_new = a, (b + 1) % n
            elif s == 1:
                R_new = ec.add(R, R)
                a_new, b_new = (a * 2) % n, (b * 2) % n
            else:
                R_new = ec.add(R, G)
                a_new, b_new = (a + 1) % n, b

            ops += 1

            # Check if distinguished
            if is_distinguished(R_new, dp_bits):
                point_key = R_new if R_new is not None else "inf"
                store_key = (target_i, point_key)

                if store_key in dp_store:
                    # Two walks for the SAME target reached the same DP
                    sa, sb = dp_store[store_key]

                    # Collision: a_new*G + b_new*Q = sa*G + sb*Q
                    # => (a_new - sa)*G = (sb - b_new)*Q
                    da = (a_new - sa) % n
                    db = (sb - b_new) % n

                    k = _try_extract_dlog(ec, G, Q_i, da, db, n)
                    if k is not None:
                        solved[target_i] = k

                dp_store[store_key] = (a_new, b_new)

            new_walks.append((R_new, a_new, b_new, target_i))

        walks = new_walks

        if len(solved) == T:
            break

    return solved, ops


def main():
    print()
    print("=" * 78)
    print("  MULTI-TARGET BATCH DLP ATTACK")
    print("  Amortizing work across T targets with Pollard rho")
    print("=" * 78)

    # Test on medium-sized curves
    test_configs = [
        (797, 1),    # single target baseline
        (797, 2),
        (797, 5),
        (797, 10),
        (797, 20),
        (1601, 1),
        (1601, 5),
        (1601, 10),
        (1601, 20),
        (3203, 1),
        (3203, 5),
        (3203, 10),
        (3203, 20),
    ]

    results = []

    for p_val, n_targets in test_configs:
        ec = SmallEC(p_val, 0, 7)
        N = ec.order
        G = ec.generator

        if G is None or N <= 2:
            continue

        group_ord = _subgroup_order(ec, G)
        max_ops = group_ord * 10

        # Generate targets using the correct subgroup order
        targets = []
        for _ in range(n_targets):
            k = secrets.randbelow(group_ord - 1) + 1
            Q = ec.multiply(G, k)
            targets.append((Q, k))

        # Single-target baseline
        if n_targets == 1:
            t0 = time.time()
            k_found, ops = single_target_rho(
                ec, G, targets[0][0], max_ops, group_order=group_ord
            )
            dt = (time.time() - t0) * 1000
            # Verify: k_found might differ from targets[0][1] but still be correct
            # (equivalent discrete logs modulo group order)
            correct = (k_found is not None and ec.multiply(G, k_found) == targets[0][0])
            print(f"\n  p={p_val}, |E|={N}, gen_order={group_ord}, T=1 (single): "
                  f"ops={ops:,d}, correct={correct}, {dt:.1f}ms")
            results.append({
                "prime": p_val, "order": N, "gen_order": group_ord,
                "n_targets": 1, "ops_total": ops, "ops_per_target": ops,
                "solved": 1 if correct else 0, "n_targets_total": 1,
            })
            continue

        # Multi-target
        t0 = time.time()
        solved, ops = multi_target_rho(
            ec, G, targets, max_ops, group_order=group_ord
        )
        dt = (time.time() - t0) * 1000

        # Count correctly solved (verify via point multiplication, not just key equality)
        n_solved = 0
        for i, (Q_i, _) in enumerate(targets):
            if i in solved and ec.multiply(G, solved[i]) == Q_i:
                n_solved += 1
        ops_per = ops / max(n_solved, 1)

        print(f"  p={p_val}, |E|={N}, gen_order={group_ord}, T={n_targets}: "
              f"ops={ops:,d}, solved={n_solved}/{n_targets}, "
              f"ops/target={ops_per:,.0f}, {dt:.1f}ms")

        results.append({
            "prime": p_val, "order": N, "gen_order": group_ord,
            "n_targets": n_targets, "ops_total": ops,
            "ops_per_target": int(ops_per),
            "solved": n_solved, "n_targets_total": n_targets,
        })

    # ================================================================
    # SCALING ANALYSIS
    # ================================================================
    print(f"\n\n{'='*78}")
    print(f"  SCALING ANALYSIS FOR BITCOIN")
    print(f"{'='*78}")

    print(f"\n  Number of exposed Bitcoin public keys (approximate):")
    print(f"    Reused addresses (P2PKH with spent outputs): ~2^30 (~1 billion)")
    print(f"    All addresses ever created: ~2^33 (~8 billion)")
    print(f"    Worst case (all EC public keys in existence): ~2^40")

    print(f"\n  Multi-target speedup for secp256k1:")
    print(f"    Single target: 2^128 operations")
    print(f"    T = 2^30 (1B keys): 2^128 / sqrt(2^30) = 2^128 / 2^15 = 2^113 ops")
    print(f"    T = 2^33 (8B keys): 2^128 / sqrt(2^33) = 2^128 / 2^16.5 = 2^111.5 ops")
    print(f"    T = 2^40 (1T keys): 2^128 / sqrt(2^40) = 2^128 / 2^20 = 2^108 ops")

    print(f"\n  Even with T = 2^40 targets:")
    print(f"    2^108 operations still takes ~10^15 YEARS")
    print(f"    (all supercomputers on Earth: ~10^18 ops/sec)")

    print(f"\n  Multi-target is a theoretical improvement but doesn't change")
    print(f"  the fundamental infeasibility. The sqrt(T) factor saves")
    print(f"  ~20 bits, but you still need 2^108 operations.")

    # ================================================================
    # SUMMARY
    # ================================================================
    print(f"\n\n{'='*78}")
    print(f"  SUMMARY")
    print(f"{'='*78}")
    print(f"""
  Multi-target Pollard rho:
  - Theoretical speedup: sqrt(T) when attacking T targets simultaneously
  - Uses distinguished points for memory-efficient collision detection
  - Real improvement: T=1M targets -> 1000x faster (saves ~10 bits)

  For Bitcoin (secp256k1):
  - ~2^30 to 2^40 exposed public keys on the blockchain
  - Multi-target reduces from 2^128 to ~2^108 operations
  - 2^108 is STILL completely infeasible (10^15 years)
  - The sqrt(T) speedup is real but insufficient

  For comparison, Shor's algorithm:
  - Doesn't benefit from multi-target (each key takes O(n^3) independently)
  - But O(n^3) = O(256^3) = 16.7M operations per key
  - Could crack all 2^30 keys in 2^30 * 16.7M = 2^54 operations
  - At 1 MHz gate speed: ~570 years for ALL Bitcoin keys
  - At 1 GHz gate speed: ~7 months for ALL Bitcoin keys
    """)
    # ================================================================
    # WRITE CSV
    # ================================================================
    csv_path = os.path.expanduser("~/Desktop/multi_target_batch.csv")
    if results:
        fieldnames = ["prime", "order", "gen_order", "n_targets",
                      "ops_total", "ops_per_target", "solved", "n_targets_total"]
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(results)
        print(f"\n  Results written to {csv_path}")

    print("=" * 78)


if __name__ == "__main__":
    main()
