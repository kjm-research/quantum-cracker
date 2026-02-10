"""Grand Unified Attack Tree -- How All secp256k1/ECDSA Attacks Connect.

Every successful attack against elliptic curve cryptography produces constraints
on the private key. This script models 15 attack vectors as constraint objects,
computes their pairwise synergies, and builds 7 realistic multi-vector attack
trees showing how constraints compound to reduce the effective search space.

The key insight: no single attack breaks a properly-implemented secp256k1 key.
But when multiple small leaks combine, the cumulative constraint can collapse
256-bit security to something feasible -- especially with quantum assistance.

References:
  - Boneh, Durfee, Frankel: "An Attack on RSA Given a Small Fraction of the
    Private Key Bits" (ASIACRYPT 1998) -- partial key exposure framework
  - Howgrave-Graham, Smart: "Lattice Attacks on Digital Signature Schemes"
    (Des. Codes Cryptogr. 2001) -- HNP lattice method
  - Kocher, Jaffe, Jun: "Differential Power Analysis" (CRYPTO 1999)
  - Boneh, DeMillo, Lipton: "On the Importance of Checking Cryptographic
    Protocols for Faults" (EUROCRYPT 1997) -- Bellcore fault attack
  - Grover: "A Fast Quantum Mechanical Algorithm for Database Search" (1996)
  - Jancar et al: "Minerva: The curse of ECDSA nonces" (TCHES 2020)
"""

import csv
import math
import os
import secrets
import sys
import time
from dataclasses import dataclass, field as dc_field

# ================================================================
# secp256k1 Constants
# ================================================================

SECP256K1_P = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
SECP256K1_N = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
SECP256K1_BITS = 256

CSV_ROWS_TREE = []
CSV_ROWS_MATRIX = []

# ================================================================
# Utility functions
# ================================================================

def separator(char="=", width=78):
    print(char * width)

def section_header(part_num, title):
    print()
    separator()
    print(f"  PART {part_num}: {title}")
    separator()

def log2_str(val):
    """Format a large number as 2^N."""
    if val <= 0:
        return "0"
    exp = math.log2(val) if val > 0 else 0
    if exp == int(exp):
        return f"2^{int(exp)}"
    return f"2^{exp:.1f}"


# ================================================================
# SmallEC -- small curve arithmetic for demonstrations
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

    def neg(self, P):
        if P is None: return None
        return (P[0], (self.p - P[1]) % self.p)


def find_prime_order_curves(min_order=50, max_p=500, count=3):
    """Find small curves with prime order for demonstrations."""
    def is_prime(n):
        if n < 2: return False
        if n < 4: return True
        if n % 2 == 0 or n % 3 == 0: return False
        i = 5
        while i * i <= n:
            if n % i == 0 or n % (i + 2) == 0: return False
            i += 6
        return True

    curves = []
    for p in range(23, max_p):
        if not is_prime(p): continue
        for a in range(p):
            for b in range(p):
                if (4 * a * a * a + 27 * b * b) % p == 0: continue
                ec = SmallEC(p, a, b)
                n = ec.order
                if n >= min_order and is_prime(n):
                    curves.append(ec)
                    if len(curves) >= count:
                        return curves
    return curves


def ecdsa_sign(ec, G, n, privkey, msg_hash, k=None):
    """ECDSA sign with optional explicit nonce."""
    for _ in range(100):
        if k is None:
            k_use = secrets.randbelow(n - 1) + 1
        else:
            k_use = k
        if math.gcd(k_use, n) != 1:
            if k is not None:
                return None  # caller-specified k not invertible
            continue
        R = ec.multiply(G, k_use)
        if R is None: continue
        r = R[0] % n
        if r == 0: continue
        k_inv = pow(k_use, -1, n)
        s = (k_inv * (msg_hash + r * privkey)) % n
        if s == 0: continue
        return r, s, k_use
    raise ValueError("Signing failed")


def bsgs(ec, G, n, Q, lo=0, hi=None):
    """Baby-step giant-step DLP solver: find k such that Q = kG."""
    if hi is None:
        hi = n
    size = hi - lo
    m = int(math.isqrt(size)) + 1
    # Baby steps: store j*G for j in [0, m)
    baby = {}
    step = None
    for j in range(m):
        pt = ec.multiply(G, lo + j) if lo + j > 0 else None
        if lo + j == 0:
            pt = None
        else:
            pt = ec.multiply(G, lo + j)
        key = pt if pt is None else (pt[0], pt[1])
        baby[key] = lo + j
    # Giant step: -m*G
    mG = ec.multiply(G, m)
    neg_mG = ec.neg(mG)
    gamma = Q
    for i in range(m):
        key = gamma if gamma is None else (gamma[0], gamma[1])
        if key in baby:
            return baby[key] + i * m
        gamma = ec.add(gamma, neg_mG)
    return None


# ================================================================
# AttackConstraint data model
# ================================================================

@dataclass
class AttackConstraint:
    name: str
    short: str           # 3-letter code for matrix
    category: str        # entropy, bit_leak, algebraic, quantum
    bits_revealed: int   # how many of 256 bits this constrains
    constraint_type: str # bits_known, search_reduction, full_recovery, algebraic
    prerequisites: str   # what access is needed
    cost: str            # human-readable cost
    feeds_into: list     # attack names this enables
    description: str = ""

ALL_ATTACKS = [
    # -- Entropy Reduction --
    AttackConstraint(
        "brain_wallet", "BRN", "entropy", 236,
        "search_reduction", "passphrase dictionary",
        "$0 (GPU + wordlist)", ["bsgs_classical", "grover_hybrid"],
        "SHA256(passphrase) -> ~20 bits effective entropy"),
    AttackConstraint(
        "sequential_keys", "SEQ", "entropy", 225,
        "search_reduction", "knowledge of key gen method",
        "$0 (CPU)", ["bsgs_classical"],
        "Key = small integer/timestamp -> ~31 bits"),
    AttackConstraint(
        "profanity_vanity", "VAN", "entropy", 224,
        "search_reduction", "knowledge of vanity tool",
        "$0 (GPU)", ["bsgs_classical"],
        "Profanity PRNG -> 32-bit seed space"),
    # -- Bit Leakage --
    AttackConstraint(
        "timing_side_channel", "TIM", "bit_leak", 15,
        "bits_known", "network timing of signing",
        "$0 (remote network)", ["nonce_bias_lattice", "bsgs_classical"],
        "Leaks nonce bit-length -> Hamming weight ~ 10-20 bits"),
    AttackConstraint(
        "dpa_power_analysis", "DPA", "bit_leak", 256,
        "full_recovery", "physical proximity + EM probe",
        "$10K (oscilloscope + probe)", ["bsgs_classical"],
        "100% bit recovery with 500 traces at realistic noise"),
    AttackConstraint(
        "fault_injection", "FLT", "bit_leak", 256,
        "full_recovery", "physical access to signing device",
        "$50K (laser/glitch equipment)", [],
        "Bellcore: 1 faulty sig -> algebraic key recovery"),
    AttackConstraint(
        "cold_boot_ram", "RAM", "bit_leak", 230,
        "bits_known", "physical access + LN2",
        "$50 (LN2 canister)", ["bsgs_classical", "grover_hybrid"],
        "RAM remanence: 90-99.9% bits intact after cooling"),
    AttackConstraint(
        "cache_side_channel", "CSC", "bit_leak", 20,
        "bits_known", "co-located process",
        "$0 (shared server)", ["bsgs_classical", "grover_hybrid"],
        "Cache timing: ~20 bits of access pattern leakage"),
    AttackConstraint(
        "em_emanations", "EME", "bit_leak", 30,
        "bits_known", "physical proximity (1m)",
        "$5K (SDR + antenna)", ["bsgs_classical", "grover_hybrid"],
        "EM emissions during signing: ~30 bits"),
    # -- Algebraic --
    AttackConstraint(
        "nonce_reuse", "NRU", "algebraic", 256,
        "full_recovery", "2 signatures with same nonce",
        "$0 (blockchain scan)", [],
        "Same k in 2 sigs: k = (h1-h2)/(s1-s2), then d = (sk-h)/r"),
    AttackConstraint(
        "nonce_bias_lattice", "LAT", "algebraic", 256,
        "full_recovery", "200 biased signatures",
        "$0 (blockchain + LLL)", [],
        "1-8 bit nonce bias + LLL lattice = full key recovery"),
    AttackConstraint(
        "r_value_reuse", "RVR", "algebraic", 256,
        "full_recovery", "blockchain scan for duplicate r",
        "$0 (blockchain scan)", [],
        "Identical r values = nonce reuse = instant recovery"),
    # -- Quantum --
    AttackConstraint(
        "bsgs_classical", "BSG", "quantum", 0,
        "search_reduction", "classical CPU/GPU",
        "$variable", [],
        "O(2^((256-known)/2)) -- uses partial info from other attacks"),
    AttackConstraint(
        "grover_hybrid", "GRV", "quantum", 0,
        "search_reduction", "quantum computer",
        "$variable + qubits", [],
        "O(2^((256-known)/2)) with sqrt quantum speedup on remainder"),
    AttackConstraint(
        "shor_full", "SHR", "quantum", 256,
        "full_recovery", "2330 logical qubits",
        "$billions (quantum computer)", [],
        "O(n^3) -- breaks ECDLP completely, hardware doesn't exist yet"),
]

ATTACK_MAP = {a.name: a for a in ALL_ATTACKS}


# ================================================================
# Part 1: Attack Constraint Model
# ================================================================

def part1_constraint_model():
    section_header(1, "ATTACK CONSTRAINT MODEL (15 vectors)")
    print()
    print("  Each attack produces constraints on the 256-bit private key.")
    print("  Constraints combine to shrink the effective search space.")
    print()

    for cat in ["entropy", "bit_leak", "algebraic", "quantum"]:
        label = {"entropy": "ENTROPY REDUCTION", "bit_leak": "BIT LEAKAGE",
                 "algebraic": "ALGEBRAIC RELATIONS", "quantum": "SEARCH METHODS"}[cat]
        print(f"  [{label}]")
        print(f"  {'Name':<25} {'Bits':>4}  {'Type':<18} {'Prerequisites':<30}")
        print(f"  {'-'*25} {'----':>4}  {'-'*18} {'-'*30}")
        for a in ALL_ATTACKS:
            if a.category != cat: continue
            bits_str = str(a.bits_revealed) if a.bits_revealed > 0 else "var"
            print(f"  {a.name:<25} {bits_str:>4}  {a.constraint_type:<18} {a.prerequisites:<30}")
        print()

    print("  KEY INSIGHT: Attacks in different categories are ORTHOGONAL --")
    print("  they constrain different bits, so their effects multiply.")
    print()


# ================================================================
# Part 2: Combination Matrix
# ================================================================

def compute_compatibility(a, b):
    """Determine relationship between two attacks."""
    if a.name in b.feeds_into or b.name in a.feeds_into:
        rel = "enables"
        combined = min(a.bits_revealed + b.bits_revealed, 256)
        synergy = 2.0  # strong synergy
    elif a.category == b.category and a.category in ("bit_leak",):
        if a.constraint_type == "full_recovery" or b.constraint_type == "full_recovery":
            rel = "redundant"
            combined = max(a.bits_revealed, b.bits_revealed)
            synergy = 0.5
        else:
            rel = "independent"
            combined = min(a.bits_revealed + b.bits_revealed, 256)
            synergy = 1.5
    elif a.category == "algebraic" and b.category == "algebraic":
        rel = "redundant"
        combined = max(a.bits_revealed, b.bits_revealed)
        synergy = 0.3
    elif a.category in ("quantum",) and b.category in ("quantum",):
        rel = "compatible"
        combined = max(a.bits_revealed, b.bits_revealed)
        synergy = 0.5
    else:
        rel = "independent"
        combined = min(a.bits_revealed + b.bits_revealed, 256)
        synergy = 1.0 + (a.bits_revealed + b.bits_revealed) / 256

    return {
        "relationship": rel,
        "combined_bits": combined,
        "synergy": round(synergy, 2),
    }


def part2_combination_matrix():
    section_header(2, "ATTACK COMBINATION MATRIX (15 x 15)")
    print()
    print("  Codes: E=enables  I=independent  R=redundant  C=compatible")
    print()

    codes = {"enables": "E", "independent": "I", "redundant": "R", "compatible": "C"}

    # Header row
    header = "          " + " ".join(f"{a.short}" for a in ALL_ATTACKS)
    print(f"  {header}")
    print(f"  {'':>9}" + " ---" * len(ALL_ATTACKS))

    pairs = []
    for i, a in enumerate(ALL_ATTACKS):
        row = f"  {a.short:>6}  |"
        for j, b in enumerate(ALL_ATTACKS):
            if i == j:
                row += "  . "
            else:
                comp = compute_compatibility(a, b)
                row += f"  {codes[comp['relationship']]} "
                if i < j:
                    pairs.append((a, b, comp))
                    CSV_ROWS_MATRIX.append({
                        "attack_a": a.name, "attack_b": b.name,
                        "compatible": comp["relationship"],
                        "combined_bits": comp["combined_bits"],
                        "synergy_score": comp["synergy"],
                        "notes": f"{a.description[:40]}+{b.description[:40]}"
                    })
        print(row)

    # Top synergistic pairs
    pairs.sort(key=lambda x: x[2]["synergy"], reverse=True)
    print()
    print("  TOP 10 SYNERGISTIC COMBINATIONS:")
    print(f"  {'#':>3} {'Attack A':<25} {'Attack B':<25} {'Combined':>8} {'Synergy':>8}")
    print(f"  {'-'*3} {'-'*25} {'-'*25} {'-'*8} {'-'*8}")
    for i, (a, b, comp) in enumerate(pairs[:10]):
        print(f"  {i+1:>3} {a.name:<25} {b.name:<25} {comp['combined_bits']:>5}/256 {comp['synergy']:>7.2f}")
    print()


# ================================================================
# Part 3: Seven Attack Trees
# ================================================================

def tree1_the_insider():
    """Cold boot attack -> BSGS on remaining bits."""
    print()
    print("  TREE 1: \"The Insider\" (physical access to signing device)")
    print("  " + "-" * 60)
    print()
    print("  Scenario: Attacker freezes RAM with LN2, reads memory contents.")
    print("  At -50C, 99.9% of DRAM bits retain value for 10+ minutes.")
    print()
    print("  Step 1: Cold boot -> recover 230/256 bits (0.1% error rate)")
    print("  Step 2: BSGS on remaining 26 unknown bits")
    print("         Search space: 2^26 = 67 million")
    print("         BSGS cost: sqrt(2^26) = 2^13 = 8,192 operations")
    print()

    # Small-curve demo: know top bits, BSGS on bottom 3
    curves = find_prime_order_curves(min_order=80, max_p=300, count=1)
    ec = curves[0]
    G = ec.generator
    n = ec.order
    privkey = secrets.randbelow(n - 2) + 1
    Q = ec.multiply(G, privkey)

    key_bits = privkey.bit_length()
    known_count = max(key_bits - 3, 0)
    known_mask = (privkey >> 3) << 3  # know all but bottom 3 bits
    unknown_range = 8  # 2^3

    print(f"  Small-curve demo (p={ec.p}, n={n}):")
    print(f"    Private key: {privkey}")
    print(f"    Known bits: top {known_count} (mask = {known_mask})")
    print(f"    Unknown: bottom 3 bits (8 candidates)")

    recovered = None
    ops = 0
    for guess in range(unknown_range):
        ops += 1
        candidate = known_mask | guess
        if candidate >= n:
            continue
        if ec.multiply(G, candidate) == Q:
            recovered = candidate
            break

    if recovered == privkey:
        print(f"    RECOVERED: {recovered} in {ops} operations")
    else:
        print(f"    Recovery failed (unexpected)")

    print()
    print("  secp256k1 projection:")
    print("    230 bits from RAM -> 26 unknown -> BSGS 2^13 = instant")
    print("    Even at 10% error (205 bits): BSGS 2^25 ~ 30 seconds")
    print("    At 35% error (166 bits): BSGS 2^45 ~ years (need Grover)")
    print("    Feasibility: TODAY with physical access + $50 of LN2")
    print()

    for error_rate, bits_known, label in [
        (0.001, 230, "LN2 cooled"), (0.01, 205, "quick freeze"),
        (0.1, 166, "room temp 5min"), (0.35, 90, "warm RAM")]:
        remaining = SECP256K1_BITS - bits_known
        bsgs_cost = 2 ** (remaining // 2)
        CSV_ROWS_TREE.append({
            "tree_name": "1-Insider", "step_number": 1,
            "attack_name": f"cold_boot ({label})",
            "bits_revealed": bits_known,
            "cumulative_bits_known": bits_known,
            "remaining_search_space": f"2^{remaining}",
            "classical_cost": f"2^{remaining//2}",
            "quantum_cost": f"2^{remaining//4}" if remaining > 20 else "trivial",
            "qubits_needed": max(remaining * 2, 0) if remaining > 20 else 0,
            "feasibility_year": "NOW" if remaining <= 40 else "2030+" if remaining <= 160 else "2040+"
        })


def tree2_the_network_observer():
    """Minerva timing -> biased nonces -> lattice recovery."""
    print()
    print("  TREE 2: \"The Network Observer\" (passive signature analysis)")
    print("  " + "-" * 60)
    print()
    print("  Scenario: Signing device has non-constant-time EC multiply.")
    print("  Attacker measures network timing for 200+ signing operations.")
    print("  Nonce bit-length leaks through timing -> Minerva attack.")
    print()
    print("  Step 1: Timing measurements -> leak nonce bit-length (~2-8 bits)")
    print("  Step 2: Collect 200 signatures with known nonce bias")
    print("  Step 3: Build HNP lattice (dim 202), run LLL")
    print("  Step 4: Shortest vector encodes private key")
    print()

    # Demo: biased nonces on small curve -> brute force recovery
    curves = find_prime_order_curves(min_order=40, max_p=300, count=1)
    ec = curves[0]
    G = ec.generator
    n = ec.order

    privkey = secrets.randbelow(n - 2) + 1
    Q = ec.multiply(G, privkey)

    # Generate signatures with biased nonces (top 2 bits = 0)
    sigs = []
    for i in range(20):
        msg = (i * 137 + 42) % n
        if msg == 0: msg = 1
        max_k = max(2, n // 4)  # top 2 bits zero
        k = secrets.randbelow(max_k - 1) + 1
        result = ecdsa_sign(ec, G, n, privkey, msg, k=k)
        if result is None: continue
        r_sig, s_sig, k_actual = result
        sigs.append((msg, r_sig, s_sig, k_actual))
        if len(sigs) >= 10: break

    print(f"  Small-curve demo (p={ec.p}, n={n}):")
    print(f"    Private key: {privkey}")
    print(f"    Biased nonces (top 2 bits = 0): k < {n//4}")
    print(f"    Collected {len(sigs)} signatures")

    # Brute force over reduced nonce space (simulating lattice on small curve)
    recovered = None
    for msg, r, s, _ in sigs[:1]:  # use first signature
        for k_guess in range(1, n // 4):
            R_test = ec.multiply(G, k_guess)
            if R_test and R_test[0] % n == r:
                d_cand = ((s * k_guess - msg) * pow(r, -1, n)) % n
                if ec.multiply(G, d_cand) == Q:
                    recovered = d_cand
                    break
        if recovered:
            break

    if recovered == privkey:
        print(f"    RECOVERED: {recovered} via biased-nonce search")
    else:
        print(f"    (Brute force on reduced space -- real attack uses LLL)")

    print()
    print("  secp256k1 projection:")
    print("    Minerva: 2-bit nonce bias -> 200 sigs -> LLL -> full key")
    print("    LadderLeak: 1-bit bias -> 500 sigs -> LLL -> full key")
    print("    LLL runs in polynomial time O(n^6 * log(max_entry))")
    print("    Feasibility: TODAY for non-constant-time implementations")
    print()

    CSV_ROWS_TREE.append({
        "tree_name": "2-NetworkObserver", "step_number": 1,
        "attack_name": "timing_nonce_bias",
        "bits_revealed": 2, "cumulative_bits_known": 2,
        "remaining_search_space": "N/A (algebraic)",
        "classical_cost": "LLL O(n^6)",
        "quantum_cost": "N/A (classical suffices)",
        "qubits_needed": 0,
        "feasibility_year": "NOW (non-constant-time impls)"
    })
    CSV_ROWS_TREE.append({
        "tree_name": "2-NetworkObserver", "step_number": 2,
        "attack_name": "lattice_hnp_recovery",
        "bits_revealed": 256, "cumulative_bits_known": 256,
        "remaining_search_space": "0",
        "classical_cost": "polynomial",
        "quantum_cost": "N/A",
        "qubits_needed": 0,
        "feasibility_year": "NOW"
    })


def tree3_the_quantum_hybrid():
    """Side channels leak bits -> Grover finishes the job."""
    print()
    print("  TREE 3: \"The Quantum Hybrid\" (side channels + near-term quantum)")
    print("  " + "-" * 60)
    print()
    print("  Scenario: Combine multiple classical side-channel leaks,")
    print("  then use a near-term quantum computer for the remaining bits.")
    print("  This DRAMATICALLY reduces quantum hardware requirements.")
    print()
    print("  Step 1: Timing side-channel -> 15 bits (Hamming weight)")
    print("  Step 2: Cache side-channel -> 20 bits (access pattern)")
    print("  Step 3: EM emanations    -> 30 bits (radiation)")
    print("  Step 4: Partial memory   -> 35 bits (swap/dump)")
    print("  Total classical leakage: 100 bits")
    print("  Remaining: 156 unknown bits")
    print()
    print("  Step 5: Grover search on 156 bits")
    print("         Classical BSGS: 2^78 = infeasible")
    print("         Grover: 2^78 iterations but only ~160 logical qubits")
    print("         Compare: full Shor needs 2330 qubits")
    print()

    # Demo: know 60% of a small-curve key, Grover-simulate the rest
    curves3 = find_prime_order_curves(min_order=80, max_p=300, count=1)
    ec = curves3[0]
    G = ec.generator
    n = ec.order

    privkey = secrets.randbelow(n - 2) + 1
    Q = ec.multiply(G, privkey)
    key_bits = n.bit_length()
    known_frac = 0.6
    known_count = int(key_bits * known_frac)
    unknown_count = key_bits - known_count

    # Simulate knowing top 60% of bits
    shift = unknown_count
    known_part = (privkey >> shift) << shift
    unknown_part = privkey & ((1 << shift) - 1)
    search_space = 1 << shift

    print(f"  Small-curve demo (p={ec.p}, n={n}, key_bits={key_bits}):")
    print(f"    Private key: {privkey}")
    print(f"    Known: top {known_count} bits (mask = {known_part})")
    print(f"    Unknown: bottom {unknown_count} bits (search space = {search_space})")

    # Classical brute force (simulating what Grover would do)
    grover_iters = int(math.pi / 4 * math.sqrt(search_space))
    ops = 0
    for guess in range(search_space):
        ops += 1
        candidate = known_part | guess
        if candidate >= n: continue
        if ec.multiply(G, candidate) == Q:
            print(f"    RECOVERED: {candidate} in {ops} ops (Grover would: ~{grover_iters} ops)")
            break
    else:
        print(f"    Searched {ops} candidates")

    print()
    print("  secp256k1 projection:")
    print(f"    100 bits known -> 156 remaining")
    print(f"    Classical BSGS: 2^78 operations = infeasible")
    print(f"    Grover: ~2^78 iterations with ~312 logical qubits")
    print(f"    140 bits known -> 116 remaining -> 2^58 Grover, ~232 qubits")
    print(f"    200 bits known -> 56 remaining -> 2^28 Grover, ~112 qubits")
    print(f"    236 bits known -> 20 remaining -> 2^10 Grover, ~40 qubits")
    print()
    print("  The MORE classical info you leak, the FEWER qubits you need:")
    print()
    print(f"  {'Classical bits':>16} {'Remaining':>10} {'Grover iters':>14} {'Qubits':>8} {'Timeline':>10}")
    print(f"  {'-'*16} {'-'*10} {'-'*14} {'-'*8} {'-'*10}")

    for known, year in [(0, "2050+"), (50, "2045"), (100, "2035"),
                         (140, "2032"), (176, "2030"), (200, "2028"),
                         (236, "NOW*"), (246, "NOW*")]:
        remaining = SECP256K1_BITS - known
        grover_it = f"2^{remaining//2}" if remaining > 0 else "1"
        qubits = remaining * 2 if remaining > 0 else 0
        CSV_ROWS_TREE.append({
            "tree_name": "3-QuantumHybrid", "step_number": known,
            "attack_name": f"grover_with_{known}_known_bits",
            "bits_revealed": known, "cumulative_bits_known": known,
            "remaining_search_space": f"2^{remaining}",
            "classical_cost": f"2^{remaining//2}",
            "quantum_cost": grover_it,
            "qubits_needed": qubits,
            "feasibility_year": year
        })
        print(f"  {known:>16} {remaining:>10} {grover_it:>14} {qubits:>8} {year:>10}")

    print()
    print("  *With Grover on 20 bits, even today's ~20 logical qubits suffice")
    print()


def tree4_the_lazy_developer():
    """Weak PRNG -> brute force seed -> algebraic key recovery."""
    print()
    print("  TREE 4: \"The Lazy Developer\" (weak RNG implementation)")
    print("  " + "-" * 60)
    print()
    print("  Scenario: Signing software uses 32-bit PRNG state for nonces.")
    print("  Attacker brute-forces the PRNG seed from one signature.")
    print()
    print("  Step 1: Observe one (r, s) signature on blockchain")
    print("  Step 2: For each 32-bit seed, generate candidate nonce k")
    print("  Step 3: Check if k produces observed r value")
    print("  Step 4: Recover private key algebraically: d = (sk - h) / r")
    print()

    curves = find_prime_order_curves(min_order=40, max_p=300, count=1)
    ec = curves[0]
    G = ec.generator
    n = ec.order
    privkey = secrets.randbelow(n - 2) + 1
    Q = ec.multiply(G, privkey)

    # Simulate weak PRNG: nonce = (seed * 137 + 43) % n
    seed = secrets.randbelow(256)  # 8-bit seed for demo
    weak_k = (seed * 137 + 43) % (n - 1) + 1
    msg = 42 % n or 1
    result = ecdsa_sign(ec, G, n, privkey, msg, k=weak_k)
    if result is None:
        weak_k = (seed * 137 + 47) % (n - 1) + 1
        result = ecdsa_sign(ec, G, n, privkey, msg, k=weak_k)
    r, s, _ = result

    print(f"  Small-curve demo (p={ec.p}, n={n}):")
    print(f"    Private key: {privkey}, weak seed: {seed}")
    print(f"    Signature: r={r}, s={s}")

    # Brute force the seed
    recovered = None
    for seed_guess in range(256):
        k_guess = (seed_guess * 137 + 43) % (n - 1) + 1
        if math.gcd(k_guess, n) != 1: continue
        R_test = ec.multiply(G, k_guess)
        if R_test and R_test[0] % n == r:
            d_cand = ((s * k_guess - msg) * pow(r, -1, n)) % n
            if ec.multiply(G, d_cand) == Q:
                recovered = d_cand
                break

    if recovered == privkey:
        print(f"    RECOVERED: key={recovered} from seed={seed_guess}")
    print()
    print("  secp256k1 projection:")
    print("    32-bit PRNG seed: 2^32 = 4.3 billion candidates")
    print("    At 10^9 checks/sec (GPU): ~4 seconds")
    print("    Real world: Profanity vanity gen -> $160M Wintermute hack (2022)")
    print("    Feasibility: INSTANT on commodity hardware")
    print()

    CSV_ROWS_TREE.append({
        "tree_name": "4-LazyDeveloper", "step_number": 1,
        "attack_name": "weak_prng_brute_force",
        "bits_revealed": 224, "cumulative_bits_known": 224,
        "remaining_search_space": "2^32",
        "classical_cost": "2^32 (~4 sec GPU)",
        "quantum_cost": "N/A",
        "qubits_needed": 0,
        "feasibility_year": "NOW"
    })


def tree5_the_perfect_storm():
    """Multiple independent leaks compound."""
    print()
    print("  TREE 5: \"The Perfect Storm\" (5 independent leaks compound)")
    print("  " + "-" * 60)
    print()
    print("  Scenario: Target has MULTIPLE small leaks. Individually harmless,")
    print("  but they constrain DIFFERENT bits of the key (orthogonal info).")
    print()

    leaks = [
        ("Timing (Hamming weight)", 15, "bits 241-255 (MSB structure)"),
        ("Cache (access pattern)", 20, "bits 200-219 (lookup indices)"),
        ("EM (emanations)", 30, "bits 160-189 (multiply pattern)"),
        ("Memory (partial dump)", 35, "bits 100-134 (page boundary)"),
        ("Power (DPA partial)", 15, "bits 60-74 (final add steps)"),
    ]

    total = sum(bits for _, bits, _ in leaks)
    remaining = SECP256K1_BITS - total

    cumulative = 0
    for name, bits, where in leaks:
        cumulative += bits
        rem = SECP256K1_BITS - cumulative
        print(f"  Leak: {name:<30} {bits:>3} bits ({where})")
        print(f"        Cumulative: {cumulative}/256 known, {rem} remaining")
        print(f"        Classical:  2^{rem//2} BSGS | Quantum: 2^{rem//2} Grover @ {rem*2} qubits")
        CSV_ROWS_TREE.append({
            "tree_name": "5-PerfectStorm", "step_number": cumulative,
            "attack_name": name.split("(")[0].strip().lower(),
            "bits_revealed": bits, "cumulative_bits_known": cumulative,
            "remaining_search_space": f"2^{rem}",
            "classical_cost": f"2^{rem//2}",
            "quantum_cost": f"2^{rem//2}",
            "qubits_needed": rem * 2,
            "feasibility_year": "NOW" if rem <= 40 else "2030" if rem <= 160 else "2040+"
        })
    print()
    print(f"  COMBINED: {total} bits known, {remaining} remaining")
    print(f"    Classical BSGS:  2^{remaining//2} = {'feasible' if remaining <= 80 else 'infeasible'}")
    print(f"    Grover hybrid:   2^{remaining//2} iterations @ {remaining*2} qubits")
    print()

    # Small-curve demo
    curves5 = find_prime_order_curves(min_order=80, max_p=300, count=1)
    ec = curves5[0]
    G = ec.generator
    n = ec.order
    privkey = secrets.randbelow(n - 2) + 1
    Q = ec.multiply(G, privkey)
    key_bits = n.bit_length()

    # Simulate 5 independent leaks revealing different bit ranges
    known_bits_set = set()
    bits_per_leak = key_bits // 5
    for i in range(5):
        start = i * bits_per_leak
        end = min(start + bits_per_leak, key_bits)
        # Reveal 60% of bits in each range
        reveal_count = max(1, int((end - start) * 0.6))
        positions = list(range(start, end))[:reveal_count]
        known_bits_set.update(positions)

    # Build candidate from known bits
    known_mask = 0
    for pos in known_bits_set:
        if privkey & (1 << pos):
            known_mask |= (1 << pos)

    unknown_positions = [i for i in range(key_bits) if i not in known_bits_set]
    unknown_count = len(unknown_positions)
    search_size = 1 << unknown_count

    print(f"  Small-curve demo (p={ec.p}, n={n}, key_bits={key_bits}):")
    print(f"    Key: {privkey}, known bits: {len(known_bits_set)}/{key_bits}")
    print(f"    Unknown positions: {unknown_count} -> search space: {search_size}")

    recovered = None
    ops = 0
    for guess_val in range(min(search_size, 100000)):
        ops += 1
        candidate = known_mask
        for bit_idx, pos in enumerate(unknown_positions):
            if guess_val & (1 << bit_idx):
                candidate |= (1 << pos)
        if candidate >= n: continue
        if candidate == 0: continue
        if ec.multiply(G, candidate) == Q:
            recovered = candidate
            break

    if recovered == privkey:
        print(f"    RECOVERED in {ops} ops (search space was {search_size})")
    elif ops >= 100000:
        print(f"    Demo limited to 100K ops (full search: {search_size})")
    print()


def tree6_the_blockchain_analyst():
    """Pure on-chain analysis: r-reuse, brain wallets, short nonces."""
    print()
    print("  TREE 6: \"The Blockchain Analyst\" (pure on-chain scavenging)")
    print("  " + "-" * 60)
    print()
    print("  Scenario: No physical access, no side channels. Just the")
    print("  blockchain. Scan for signatures revealing implementation flaws.")
    print()

    curves = find_prime_order_curves(min_order=40, max_p=300, count=1)
    ec = curves[0]
    G = ec.generator
    n = ec.order
    privkey = secrets.randbelow(n - 2) + 1
    Q = ec.multiply(G, privkey)

    # Attack A: r-value reuse (same nonce k)
    print("  Attack A: r-value reuse detection")
    shared_k = secrets.randbelow(n - 2) + 1
    msg1, msg2 = 17 % n or 1, 42 % n or 1
    r1, s1, _ = ecdsa_sign(ec, G, n, privkey, msg1, k=shared_k)
    r2, s2, _ = ecdsa_sign(ec, G, n, privkey, msg2, k=shared_k)

    assert r1 == r2, "Same k should give same r"
    # Recover k: k = (h1 - h2) / (s1 - s2)
    k_recovered = ((msg1 - msg2) * pow((s1 - s2) % n, -1, n)) % n
    d_recovered = ((s1 * k_recovered - msg1) * pow(r1, -1, n)) % n
    print(f"    Same r detected: r1=r2={r1}")
    print(f"    Recovered nonce: k={k_recovered} (actual: {shared_k})")
    print(f"    Recovered key:   d={d_recovered} (actual: {privkey})")
    print(f"    Result: {'SUCCESS' if d_recovered == privkey else 'FAILED'}")
    print()

    # Attack B: brain wallet dictionary
    print("  Attack B: brain wallet dictionary scan")
    brain_phrases = ["password", "bitcoin", "satoshi", "123456", "letmein"]
    target_phrase = "satoshi"
    import hashlib
    target_key = int(hashlib.sha256(target_phrase.encode()).hexdigest(), 16) % (n - 1) + 1
    target_pub = ec.multiply(G, target_key)

    found = False
    for phrase in brain_phrases:
        candidate = int(hashlib.sha256(phrase.encode()).hexdigest(), 16) % (n - 1) + 1
        if ec.multiply(G, candidate) == target_pub:
            print(f"    FOUND: \"{phrase}\" -> key {candidate}")
            found = True
            break
    if not found:
        print(f"    Dictionary exhausted, no match")
    print()

    # Attack C: short nonce brute force
    print("  Attack C: short nonce brute force")
    short_k = secrets.randbelow(min(50, n - 1)) + 1
    msg3 = 99 % n or 1
    result3 = ecdsa_sign(ec, G, n, privkey, msg3, k=short_k)
    while result3 is None:
        short_k = secrets.randbelow(min(50, n - 1)) + 1
        result3 = ecdsa_sign(ec, G, n, privkey, msg3, k=short_k)
    r3, s3, _ = result3

    for k_guess in range(1, 100):
        if math.gcd(k_guess, n) != 1: continue
        R_test = ec.multiply(G, k_guess)
        if R_test and R_test[0] % n == r3:
            d_cand = ((s3 * k_guess - msg3) * pow(r3, -1, n)) % n
            if ec.multiply(G, d_cand) == Q:
                print(f"    Short nonce found: k={k_guess} -> key={d_cand}")
                break
    print()

    print("  secp256k1 projection:")
    print("    r-reuse: scan 700M+ Bitcoin transactions for duplicate r values")
    print("             Known instances: hundreds of keys recovered (2011-2024)")
    print("    Brain wallet: GPU dictionary attack at 10^9 SHA256/sec")
    print("             Known thefts: sweeper bots drain within minutes")
    print("    Short nonce: scan for small k values (k < 2^32)")
    print("             Known: Bitcoin puzzle transactions keys 1-66")
    print("    Feasibility: ONGOING -- bots actively scan the mempool")
    print()

    for attack, bits in [("r_value_reuse", 256), ("brain_wallet_dict", 236), ("short_nonce_brute", 224)]:
        CSV_ROWS_TREE.append({
            "tree_name": "6-BlockchainAnalyst", "step_number": 1,
            "attack_name": attack,
            "bits_revealed": bits, "cumulative_bits_known": bits,
            "remaining_search_space": f"2^{256-bits}" if bits < 256 else "0",
            "classical_cost": "O(1)" if bits == 256 else f"2^{256-bits}",
            "quantum_cost": "N/A",
            "qubits_needed": 0,
            "feasibility_year": "NOW"
        })


def tree7_maximum_effort():
    """Nation-state escalation ladder."""
    print()
    print("  TREE 7: \"Maximum Effort\" (nation-state escalation ladder)")
    print("  " + "-" * 60)
    print()
    print("  Scenario: Well-funded adversary targets a specific HSM/wallet.")
    print("  Each attack escalation requires more resources but bypasses")
    print("  the countermeasures that blocked the previous attempt.")
    print()

    steps = [
        ("Level 1: Fault injection (Bellcore)",
         "Glitch voltage during signing -> faulty (r',s')",
         "Blocked by: verify-after-sign",
         "fault_injection", 256, "$50K", 0, "NOW (if no verify)"),
        ("Level 2: DPA (500 power traces)",
         "Oscilloscope on power rail -> bit-by-bit recovery",
         "Blocked by: scalar blinding + Montgomery ladder",
         "dpa_500_traces", 256, "$10K", 0, "NOW (if no blinding)"),
        ("Level 3: Combined side-channels",
         "Timing + cache + EM -> 100 bits -> BSGS on 156",
         "Blocked by: 2^78 BSGS is infeasible classically",
         "combined_sidechannel", 100, "$100K", 0, "infeasible alone"),
        ("Level 4: Side-channels + Grover",
         "100 bits classical + Grover on 156 -> 2^78 quantum",
         "Blocked by: need ~160 logical qubits",
         "sidechannel_grover", 256, "$100M+", 312, "2035"),
        ("Level 5: Full Shor's algorithm",
         "Quantum period-finding on secp256k1 ECDLP",
         "Blocked by: need 2330 logical qubits",
         "shor_full", 256, "$1B+", 2330, "2045+"),
    ]

    for label, method, blocker, name, bits, cost_str, qubits, year in steps:
        print(f"  {label}")
        print(f"    Method:  {method}")
        print(f"    {blocker}")
        print(f"    Cost: {cost_str}, Qubits: {qubits if qubits > 0 else 'N/A'}")
        print()

        CSV_ROWS_TREE.append({
            "tree_name": "7-MaximumEffort", "step_number": steps.index((label, method, blocker, name, bits, cost_str, qubits, year)) + 1,
            "attack_name": name,
            "bits_revealed": bits, "cumulative_bits_known": bits,
            "remaining_search_space": "0" if bits >= 256 else f"2^{256-bits}",
            "classical_cost": cost_str,
            "quantum_cost": f"2^{max(0,(256-bits)//2)}" if qubits > 0 else "N/A",
            "qubits_needed": qubits,
            "feasibility_year": year
        })

    print("  ESCALATION DECISION TREE:")
    print()
    print("    Has verify-after-sign?")
    print("    |-- NO  -> Fault injection -> DONE (key recovered)")
    print("    |-- YES -> Has scalar blinding?")
    print("              |-- NO  -> DPA (500 traces) -> DONE")
    print("              |-- YES -> Has constant-time code?")
    print("                        |-- NO  -> Timing+cache+EM -> 100 bits")
    print("                        |          -> Wait for quantum (2035)")
    print("                        |-- YES -> FULLY HARDENED")
    print("                                   -> Only path: Shor's (2045+)")
    print()
    print("  Bitcoin Core (libsecp256k1): ALL defenses active")
    print("  -> Only Shor's algorithm can break it -> need 2330 logical qubits")
    print()


def part3_attack_trees():
    section_header(3, "SEVEN ATTACK TREES")
    print()
    print("  Each tree models a realistic multi-vector attack scenario.")
    print("  Small-curve demonstrations PROVE the math works.")
    print("  secp256k1 projections show real-world feasibility.")

    tree1_the_insider()
    tree2_the_network_observer()
    tree3_the_quantum_hybrid()
    tree4_the_lazy_developer()
    tree5_the_perfect_storm()
    tree6_the_blockchain_analyst()
    tree7_maximum_effort()


# ================================================================
# Part 4: Quantitative Comparison
# ================================================================

def part4_quantitative_analysis():
    section_header(4, "QUANTITATIVE COMPARISON")
    print()
    print("  All 7 trees compared side-by-side:")
    print()

    trees = [
        ("1-Insider", "Cold boot -> BSGS", 230, "physical+LN2", "NOW"),
        ("2-Observer", "Timing -> Lattice", 256, "network timing", "NOW*"),
        ("3-Quantum", "Side-ch -> Grover", 100, "side-ch+quantum", "2035"),
        ("4-Lazy Dev", "Weak PRNG -> algebra", 224, "blockchain", "NOW"),
        ("5-Storm", "5 leaks compound", 115, "multi-vector", "2030"),
        ("6-Analyst", "On-chain scan", 256, "blockchain only", "NOW"),
        ("7-Max", "Full escalation", 256, "nation-state", "2045+"),
    ]

    header = f"  {'Tree':<14} {'Attack':<22} {'Known':>5} {'Left':>5} {'Classical':>12} {'Grover':>10} {'Qubits':>7} {'When':>6}"
    print(header)
    print("  " + "-" * len(header.strip()))

    for name, attack, bits, prereqs, year in trees:
        remaining = SECP256K1_BITS - bits
        classical = f"2^{remaining//2}" if remaining > 0 else "trivial"
        grover = f"2^{remaining//2}" if remaining > 0 else "N/A"
        qubits = remaining * 2 if remaining > 0 else 0
        print(f"  {name:<14} {attack:<22} {bits:>5} {remaining:>5} {classical:>12} {grover:>10} {qubits:>7} {year:>6}")

    print()
    print("  * Tree 2 is classical-only (lattice attack needs no quantum)")
    print()
    print("  KEY FINDING: Trees 1, 2, 4, 6 work TODAY with zero quantum.")
    print("  The quantum hybrid (Tree 3, 5) reduces qubit requirements")
    print("  from 2330 (full Shor) to 70-312 (side-channel + Grover).")
    print()


# ================================================================
# Part 5: Connection Graph
# ================================================================

def part5_connection_graph():
    section_header(5, "ATTACK CONNECTION GRAPH")
    print()
    print("  How attacks feed into each other -- the six key pipelines:")
    print()
    print("  ENTROPY REDUCTION          BIT LEAKAGE               ALGEBRAIC")
    print("  +-----------------+        +------------------+      +----------+")
    print("  | Brain wallet    |        | Timing (15 bits) |----->| Lattice  |")
    print("  | Sequential keys |        | DPA (256 bits)   |--+   | HNP      |")
    print("  | Profanity PRNG  |        | Fault (256 bits) |  |   +----------+")
    print("  | Weak RNG        |        | Cold boot (230)  |  |        |")
    print("  +-----------------+        | Cache (20 bits)  |  |        |")
    print("         |                   | EM (30 bits)     |  |        v")
    print("         |                   +------------------+  |  +----------+")
    print("         v                          |              +->| Nonce    |")
    print("  +------------------+              v                 | reuse    |")
    print("  | Reduced search   |       +-------------+         +----------+")
    print("  | space            |------>| Known bit   |")
    print("  | (brute force)    |       | mask        |         SEARCH")
    print("  +------------------+       +-------------+        +----------+")
    print("                                    |            +->| BSGS     |")
    print("                                    +------------|  | O(sqrt)  |")
    print("                                    |            |  +----------+")
    print("                                    v            |       |")
    print("                             +-------------+     |       v")
    print("                             | Combined    |-----+  +----------+")
    print("                             | constraint  |------->| Grover   |")
    print("                             +-------------+        | hybrid   |")
    print("                                                    +----------+")
    print("                                                         |")
    print("                                                         v")
    print("                                                    +----------+")
    print("                                                    | Shor's   |")
    print("                                                    | (bypass) |")
    print("                                                    +----------+")
    print()
    print("  THE SIX MATHEMATICAL PIPELINES:")
    print()
    print("  1. TIMING -> LATTICE")
    print("     Non-constant-time EC multiply leaks nonce bit-length.")
    print("     Bit-length = biased nonce. Biased nonce + LLL = full key.")
    print("     Math: k has known MSBs -> HNP lattice dim (m+2) -> LLL")
    print()
    print("  2. DPA -> BSGS")
    print("     Power analysis recovers N individual key bits.")
    print("     BSGS searches the remaining 2^((256-N)/2) space.")
    print("     Math: each DPA trace constrains one bit -> exponential reduction")
    print()
    print("  3. FAULT -> ALGEBRAIC")
    print("     One faulty signature (r', s') alongside correct (r, s).")
    print("     d = h(s-s') / (s'r - sr') mod n. Pure algebra, instant.")
    print("     Math: two equations in two unknowns (d, k) -> solve")
    print()
    print("  4. WEAK RNG -> EVERYTHING")
    print("     Reduced entropy = reduced search space for ANY method.")
    print("     32-bit PRNG state -> 2^32 candidates -> instant on GPU.")
    print("     Math: effective key space = PRNG state space")
    print()
    print("  5. PARTIAL INFO -> GROVER SPEEDUP")
    print("     Any N known bits reduces Grover from 2^128 to 2^((256-N)/2).")
    print("     Each classical bit saves half the quantum work.")
    print("     Math: Grover on database of size M takes O(sqrt(M)) queries")
    print()
    print("  6. ORTHOGONAL CONSTRAINTS COMPOUND")
    print("     Timing reveals MSB structure. Cache reveals lookup indices.")
    print("     EM reveals multiply patterns. Memory reveals stored bytes.")
    print("     DIFFERENT bits = ADDITIVE constraints = multiplicative reduction.")
    print("     Math: if A constrains bits {i} and B constrains bits {j},")
    print("           |{i} \\cap {j}| = 0 => combined reduction = 2^(|i|+|j|)")
    print()


# ================================================================
# Part 6: Quantum Timeline
# ================================================================

def part6_quantum_timeline():
    section_header(6, "QUANTUM HARDWARE TIMELINE")
    print()
    print("  When does each attack tree become feasible?")
    print()

    roadmap = [
        (2024, 127, 10, "IBM Eagle / Google Willow"),
        (2025, 1000, 15, "IBM Heron / Quantinuum H2"),
        (2026, 1200, 20, "IBM Flamingo target"),
        (2028, 10000, 100, "Industry target (error correction)"),
        (2030, 100000, 500, "Early fault-tolerant era"),
        (2033, 200000, 2000, "IBM 100K qubit target"),
        (2035, 500000, 5000, "Mature fault-tolerant"),
        (2040, 1000000, 50000, "Large-scale quantum"),
        (2050, 10000000, 500000, "Hypothetical scale"),
    ]

    tree_qubits = [
        ("Tree 1: Insider", 0, "Classical only"),
        ("Tree 2: Observer", 0, "Classical only (LLL)"),
        ("Tree 4: Lazy Dev", 0, "Classical only"),
        ("Tree 6: Analyst", 0, "Classical only"),
        ("Tree 5: Storm", 141, "5 leaks -> Grover 141"),
        ("Tree 3: Hybrid", 312, "100 bits -> Grover 156"),
        ("Tree 7: Max L4", 312, "Side-ch -> Grover 156"),
        ("Tree 7: Max L5", 2330, "Full Shor's"),
    ]

    print(f"  {'Year':>5}  {'Physical':>10}  {'Logical':>8}  {'Milestone':<35} {'Trees Unlocked'}")
    print(f"  {'-'*5}  {'-'*10}  {'-'*8}  {'-'*35} {'-'*25}")

    for year, phys, logical, milestone in roadmap:
        unlocked = [t[0] for t in tree_qubits if t[1] <= logical * 2]
        unlocked_str = ", ".join(unlocked[-3:]) if unlocked else "None"
        if len([t for t in tree_qubits if t[1] <= logical * 2]) > 3:
            unlocked_str = f"{len([t for t in tree_qubits if t[1] <= logical * 2])}/{len(tree_qubits)} trees"
        print(f"  {year:>5}  {phys:>10,}  {logical:>8,}  {milestone:<35} {unlocked_str}")

    print()
    print("  CRITICAL THRESHOLDS:")
    print("    0 qubits needed:    Trees 1, 2, 4, 6 -- classical attacks work NOW")
    print("    ~150 qubits needed: Tree 5 (Perfect Storm) -- estimated ~2030")
    print("    ~312 qubits needed: Trees 3, 7-L4 (Quantum Hybrid) -- estimated ~2033")
    print("    2330 qubits needed: Tree 7-L5 (Full Shor) -- estimated ~2040-2050")
    print()
    print("  BOTTOM LINE: Classical attacks already work against weak implementations.")
    print("  Quantum attacks against STRONG implementations need 2330+ logical qubits.")
    print("  The hybrid approach (side-channels + quantum) could cut that to ~312.")
    print()


# ================================================================
# Part 7: CSV Output and Summary
# ================================================================

def part7_csv_and_summary():
    section_header(7, "GRAND SUMMARY")
    print()

    # Write tree CSV
    tree_path = os.path.expanduser("~/Desktop/unified_attack_tree.csv")
    if CSV_ROWS_TREE:
        with open(tree_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "tree_name", "step_number", "attack_name", "bits_revealed",
                "cumulative_bits_known", "remaining_search_space",
                "classical_cost", "quantum_cost", "qubits_needed", "feasibility_year"
            ])
            writer.writeheader()
            writer.writerows(CSV_ROWS_TREE)
        print(f"  Attack tree CSV: {tree_path} ({len(CSV_ROWS_TREE)} rows)")

    # Write matrix CSV
    matrix_path = os.path.expanduser("~/Desktop/attack_combination_matrix.csv")
    if CSV_ROWS_MATRIX:
        with open(matrix_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "attack_a", "attack_b", "compatible", "combined_bits",
                "synergy_score", "notes"
            ])
            writer.writeheader()
            writer.writerows(CSV_ROWS_MATRIX)
        print(f"  Combination matrix CSV: {matrix_path} ({len(CSV_ROWS_MATRIX)} rows)")

    print()
    print("  ============================================================")
    print("  THE UNIFIED PICTURE")
    print("  ============================================================")
    print()
    print("  WHAT WORKS TODAY (zero quantum needed):")
    print("    - Weak RNG / brain wallets: instant key recovery")
    print("    - Nonce reuse on blockchain: algebraic recovery")
    print("    - Nonce bias + lattice: 200 biased sigs -> full key")
    print("    - Cold boot + BSGS: physical access -> seconds")
    print("    - DPA on unblinded implementation: 500 traces -> full key")
    print("    - Fault injection without verify-after-sign: 1 query -> full key")
    print()
    print("  WHAT THE COMBINATIONS REVEAL:")
    print("    - No single leak breaks a hardened implementation")
    print("    - But leaks from DIFFERENT sources are ORTHOGONAL")
    print("    - 5 independent 20-bit leaks = 100 known bits")
    print("    - Each known bit HALVES the quantum work needed")
    print("    - 100 classical bits: Shor needs 2330 qubits -> Grover needs 312")
    print("    - 200 classical bits: Grover needs only 112 qubits")
    print()
    print("  THE TREE STRUCTURE:")
    print("    ENTROPY -> reduces search space (top of tree)")
    print("    BIT LEAKS -> constrain specific positions (middle)")
    print("    ALGEBRAIC -> full recovery from signatures (separate branch)")
    print("    QUANTUM -> finishes what classical started (bottom)")
    print()
    print("  BITCOIN'S DEFENSE DEPTH:")
    print("    Layer 1: CSPRNG (blocks entropy attacks)")
    print("    Layer 2: RFC 6979 deterministic nonces (blocks algebraic)")
    print("    Layer 3: libsecp256k1 constant-time (blocks timing)")
    print("    Layer 4: Scalar blinding (blocks DPA)")
    print("    Layer 5: Verify-after-sign (blocks fault injection)")
    print("    Layer 6: 256-bit key space (blocks classical brute force)")
    print("    Layer 7: ??? (future quantum defense: PQ signatures)")
    print()
    print("  All 6 classical layers are active in Bitcoin Core.")
    print("  Only Layer 7 (post-quantum) remains undeployed.")
    print("  Estimated need: when logical qubits reach ~2330 (2040-2050)")
    print()


# ================================================================
# Main
# ================================================================

def main():
    separator()
    print("  GRAND UNIFIED ATTACK TREE")
    print("  How All secp256k1/ECDSA Attacks Connect")
    separator()
    print()
    print("  15 attack vectors, 7 realistic scenarios, 6 mathematical pipelines")
    print(f"  Target: secp256k1 ({SECP256K1_BITS}-bit key, {SECP256K1_N.bit_length()}-bit prime order)")
    print()

    t0 = time.time()

    part1_constraint_model()
    part2_combination_matrix()
    part3_attack_trees()
    part4_quantitative_analysis()
    part5_connection_graph()
    part6_quantum_timeline()
    part7_csv_and_summary()

    elapsed = time.time() - t0
    separator()
    print(f"  Completed in {elapsed:.1f}s")
    print(f"  CSVs written to ~/Desktop/")
    separator()


if __name__ == "__main__":
    main()
