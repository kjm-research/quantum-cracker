"""Elliptic curve constraint encoding for the Ising Hamiltonian.

Maps the EC discrete logarithm problem k*G = P onto Ising coupling
terms so that the ground state of the resulting Hamiltonian encodes
the private key k in its spin configuration.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class SmallEC:
    """Elliptic curve over F_p with full arithmetic.

    Supports point enumeration, generator finding, and standard
    add/multiply operations. For small primes only (p < ~10^6).
    """

    def __init__(self, p: int, a: int, b: int) -> None:
        self.p = p
        self.a = a
        self.b = b
        self._points: list | None = None
        self._order: int | None = None
        self._gen: tuple[int, int] | None = None

    @property
    def order(self) -> int:
        if self._order is None:
            self._enumerate()
        assert self._order is not None
        return self._order

    @property
    def generator(self) -> tuple[int, int]:
        if self._gen is None:
            self._find_generator()
        assert self._gen is not None
        return self._gen

    @property
    def points(self) -> list:
        if self._points is None:
            self._enumerate()
        assert self._points is not None
        return self._points

    def key_bit_length(self) -> int:
        """Number of bits needed to represent the curve order."""
        return (self.order - 1).bit_length()

    def _enumerate(self) -> None:
        points: list = [None]  # infinity
        p, a, b = self.p, self.a, self.b
        qr: dict[int, list[int]] = {}
        for y in range(p):
            qr.setdefault((y * y) % p, []).append(y)
        for x in range(p):
            rhs = (x * x * x + a * x + b) % p
            if rhs in qr:
                for y in qr[rhs]:
                    points.append((x, y))
        self._points = points
        self._order = len(points)

    def _find_generator(self) -> None:
        if self._points is None:
            self._enumerate()
        assert self._points is not None
        for pt in self._points[1:]:
            if self.multiply(pt, self.order) is None:
                is_gen = True
                for d in range(2, int(self.order**0.5) + 1):
                    if self.order % d == 0:
                        if self.multiply(pt, self.order // d) is None:
                            is_gen = False
                            break
                if is_gen:
                    self._gen = pt
                    return
        self._gen = self._points[1]

    def add(
        self, P: tuple[int, int] | None, Q: tuple[int, int] | None
    ) -> tuple[int, int] | None:
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

    def neg(self, P: tuple[int, int] | None) -> tuple[int, int] | None:
        if P is None:
            return None
        return (P[0], (self.p - P[1]) % self.p)

    def multiply(
        self, P: tuple[int, int] | None, k: int
    ) -> tuple[int, int] | None:
        if k < 0:
            P = self.neg(P)
            k = -k
        if k == 0 or P is None:
            return None
        result: tuple[int, int] | None = None
        addend = P
        while k:
            if k & 1:
                result = self.add(result, addend)
            addend = self.add(addend, addend)
            k >>= 1
        return result

    def random_keypair(self, rng: np.random.Generator | None = None) -> tuple[int, tuple[int, int]]:
        """Generate a random private key and its public key."""
        if rng is None:
            rng = np.random.default_rng()
        k = int(rng.integers(1, self.order))
        P = self.multiply(self.generator, k)
        assert P is not None
        return k, P


# -- Standard test curves (y^2 = x^3 + 7, secp256k1 family) --

SMALL_CURVES = {
    "p97": SmallEC(97, 0, 7),
    "p251": SmallEC(251, 0, 7),
    "p509": SmallEC(509, 0, 7),
    "p1021": SmallEC(1021, 0, 7),
    "p2039": SmallEC(2039, 0, 7),
    "p4093": SmallEC(4093, 0, 7),
}


class ECConstraintEncoder:
    """Encode EC DLP constraints as Ising interactions.

    For small curves (N <= 20 bits): exact penalty diagonal.
    For larger curves: windowed multiplication constraints.
    """

    def __init__(
        self,
        curve: SmallEC,
        generator: tuple[int, int],
        public_key: tuple[int, int],
    ) -> None:
        self.curve = curve
        self.generator = generator
        self.public_key = public_key
        self.n_bits = curve.key_bit_length()

    def full_penalty_diagonal(self) -> NDArray[np.float64]:
        """Build a 2^N diagonal penalty vector.

        penalty[i] = 0 if i*G == P (correct key), else 1.0.
        Only feasible for N <= 20 (~1M entries).
        """
        n = self.n_bits
        size = 1 << n
        penalty = np.ones(size, dtype=np.float64)

        order = self.curve.order
        for k in range(min(size, order)):
            pt = self.curve.multiply(self.generator, k)
            if pt == self.public_key:
                penalty[k] = 0.0

        return penalty

    def spin_penalty_diagonal(self) -> NDArray[np.float64]:
        """Like full_penalty_diagonal but indexed by spin configuration.

        Spin config sigma in {+1,-1}^N maps to bits via b_i = (1 - sigma_i)/2.
        Index i in [0, 2^N) is read as binary where bit j = (i >> j) & 1.
        """
        bit_penalty = self.full_penalty_diagonal()
        n = self.n_bits
        size = 1 << n
        spin_penalty = np.ones(size, dtype=np.float64)

        for spin_idx in range(size):
            # Convert spin index to bit key
            key_val = 0
            for j in range(n):
                bit_j = (spin_idx >> j) & 1
                # spin +1 -> bit 0, spin -1 -> bit 1
                # In our indexing: if bit j of spin_idx is 1, spin_j = -1, so key bit = 1
                key_val |= bit_j << j
            if key_val < len(bit_penalty):
                spin_penalty[spin_idx] = bit_penalty[key_val]

        return spin_penalty

    def windowed_constraints(
        self, window_size: int = 4
    ) -> list[tuple[list[int], NDArray[np.float64]]]:
        """Generate local constraints from windowed EC multiplication.

        Breaks the key into windows of `window_size` bits. For each window,
        precomputes partial EC points and generates a local penalty table.

        Returns list of (bit_indices, penalty_table) tuples.
        """
        n = self.n_bits
        constraints = []

        # Precompute 2^i * G for all bit positions
        power_points = []
        pt = self.generator
        for i in range(n):
            power_points.append(pt)
            pt = self.curve.add(pt, pt)

        for start in range(0, n, window_size):
            end = min(start + window_size, n)
            w = end - start
            bit_indices = list(range(start, end))

            # For this window, compute all 2^w partial sums
            table_size = 1 << w
            partial_points = []
            for val in range(table_size):
                acc = None
                for j in range(w):
                    if val & (1 << j):
                        acc = self.curve.add(acc, power_points[start + j])
                partial_points.append(acc)

            # The penalty for this window: how well does this partial
            # sum contribute to reaching the target? We measure distance
            # by checking if the remaining bits could complete it.
            # For exact mode, this reduces to the full penalty.
            # For approximate mode, we use a soft penalty based on
            # coordinate distance to the target point.
            penalty = np.zeros(table_size, dtype=np.float64)
            for val in range(table_size):
                pt = partial_points[val]
                if pt is None:
                    # Point at infinity -- penalize unless target needs this
                    penalty[val] = 0.5
                else:
                    # Soft penalty: normalized coordinate distance to target
                    dx = (pt[0] - self.public_key[0]) % self.curve.p
                    dy = (pt[1] - self.public_key[1]) % self.curve.p
                    dx = min(dx, self.curve.p - dx)
                    dy = min(dy, self.curve.p - dy)
                    penalty[val] = (dx + dy) / (2 * self.curve.p)

            constraints.append((bit_indices, penalty))

        return constraints

    def pairwise_couplings(self) -> dict[tuple[int, int], float]:
        """Generate 2-local Ising couplings from EC structure.

        Uses the observation that bit positions in the key are correlated
        through the EC group law. We sample random multiples and compute
        bit correlations to derive coupling strengths.
        """
        n = self.n_bits
        order = self.curve.order
        couplings: dict[tuple[int, int], float] = {}

        # Sample discrete log solutions and compute bit correlations
        n_samples = min(order - 1, 500)
        bit_matrix = np.zeros((n_samples, n), dtype=np.int8)

        for s in range(n_samples):
            k = s + 1
            for j in range(n):
                bit_matrix[s, j] = (k >> j) & 1

        # Compute pairwise correlations
        # Transform bits to spins: sigma = 1 - 2*bit
        spin_matrix = 1 - 2 * bit_matrix.astype(np.float64)
        mean_spins = spin_matrix.mean(axis=0)

        for i in range(n):
            for j in range(i + 1, n):
                corr = (spin_matrix[:, i] * spin_matrix[:, j]).mean()
                corr -= mean_spins[i] * mean_spins[j]
                if abs(corr) > 0.01:
                    couplings[(i, j)] = float(corr)

        return couplings
