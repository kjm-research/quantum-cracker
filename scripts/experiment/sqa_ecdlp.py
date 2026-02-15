"""SQA vs Classical MCMC -- ECDLP Scaling Experiment.

Compares Simulated Quantum Annealing (Suzuki-Trotter) against
classical methods for the elliptic curve discrete logarithm problem.

Methods compared:
  1. Random guessing (baseline)
  2. Standard Metropolis MCMC (single-spin flips)
  3. PDQM parity Glauber (pair hopping + parity suppression)
  4. SQA uniform (standard quantum annealing, no parity weighting)
  5. SQA PDQM (quantum annealing + parity-weighted J_perp)

Usage:
    python scripts/experiment/sqa_ecdlp.py
    python scripts/experiment/sqa_ecdlp.py --bits 8,12,16,20
    python scripts/experiment/sqa_ecdlp.py --bits 8,10,12 --n-trials 20 --replicas 64
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime

import numpy as np

sys.path.insert(0, "src")

from quantum_cracker.parity.dynamics import ParityDynamics
from quantum_cracker.parity.ec_constraints import make_curve
from quantum_cracker.parity.hamiltonian import ParityHamiltonian
from quantum_cracker.parity.sqa import SQAEngine
from quantum_cracker.parity.types import ParityConfig, SQASchedule


@dataclass
class TrialResult:
    n_bits: int
    method: str
    true_key: int
    extracted_key: int
    bit_match_rate: float
    key_recovered: bool
    energy_final: float
    elapsed_sec: float
    trial_idx: int
    sweeps_or_steps: int


def bit_match(k1: int, k2: int, n: int) -> float:
    xor = k1 ^ k2
    matching = n - bin(xor).count("1")
    return matching / n


def run_random(true_key: int, n_bits: int, rng: np.random.Generator) -> tuple[int, float]:
    if n_bits > 63:
        import secrets
        guess = secrets.randbelow(1 << n_bits)
    else:
        guess = int(rng.integers(0, 1 << n_bits))
    return guess, bit_match(guess, true_key, n_bits)


def run_standard_mcmc(
    h: ParityHamiltonian, true_key: int, n_sweeps: int, temp: float,
    rng: np.random.Generator,
) -> tuple[int, float, float]:
    n = h.n_spins
    sigma0 = rng.choice([-1, 1], size=n).astype(np.int8)
    dyn = ParityDynamics(h, h.config)
    snaps = dyn.evolve_standard_mcmc(
        sigma0, n_sweeps=n_sweeps, temperature=temp,
        target_key=true_key, log_interval=n_sweeps, rng=rng,
    )
    final_key = ParityHamiltonian.spins_to_key(snaps[-1].spins)
    return final_key, bit_match(final_key, true_key, n), snaps[-1].energy


def run_parity_glauber(
    h: ParityHamiltonian, config: ParityConfig, true_key: int,
    n_sweeps: int, temp: float, rng: np.random.Generator,
) -> tuple[int, float, float]:
    n = h.n_spins
    sigma0 = rng.choice([-1, 1], size=n).astype(np.int8)
    dyn = ParityDynamics(h, config)
    snaps = dyn.evolve_glauber(
        sigma0, n_sweeps=n_sweeps, temperature=temp,
        target_key=true_key, log_interval=n_sweeps, rng=rng,
    )
    final_key = ParityHamiltonian.spins_to_key(snaps[-1].spins)
    return final_key, bit_match(final_key, true_key, n), snaps[-1].energy


def run_sqa(
    h: ParityHamiltonian, config: ParityConfig, true_key: int,
    n_steps: int, n_replicas: int, parity_weighted: bool,
    rng: np.random.Generator,
) -> tuple[int, float, float, float]:
    """Run SQA and return (key, bit_match, energy, agreement)."""
    engine = SQAEngine(h, config)
    schedule = SQASchedule(
        n_steps=n_steps,
        gamma_initial=4.0,
        gamma_final=0.01,
        beta_initial=0.5,
        beta_final=20.0,
        n_replicas=n_replicas,
        parity_weighted=parity_weighted,
    )
    result = engine.anneal(schedule, target_key=true_key, rng=rng)
    return (
        result.extracted_key,
        result.bit_match_rate or 0.0,
        result.best_energy,
        result.replica_agreement,
    )


def main():
    parser = argparse.ArgumentParser(description="SQA vs Classical ECDLP Experiment")
    parser.add_argument("--bits", type=str, default="8,10,12,16,20,24,32",
                        help="Comma-separated target bit sizes")
    parser.add_argument("--n-trials", type=int, default=10, help="Trials per method")
    parser.add_argument("--base-sweeps", type=int, default=200, help="Base MCMC sweeps")
    parser.add_argument("--sqa-steps", type=int, default=300, help="SQA annealing steps")
    parser.add_argument("--replicas", type=int, default=32, help="Trotter replicas")
    parser.add_argument("--temperature", type=float, default=0.2, help="MCMC temperature")
    parser.add_argument("--delta-e", type=float, default=2.0, help="Parity energy gap")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    bit_sizes = [int(b.strip()) for b in args.bits.split(",")]
    rng = np.random.default_rng(args.seed)
    all_results: list[TrialResult] = []

    print("=" * 78)
    print(" SQA vs CLASSICAL -- ECDLP SCALING EXPERIMENT")
    print("=" * 78)
    print(f"  Bit sizes:      {bit_sizes}")
    print(f"  Trials/method:  {args.n_trials}")
    print(f"  Base sweeps:    {args.base_sweeps}")
    print(f"  SQA steps:      {args.sqa_steps}")
    print(f"  Trotter P:      {args.replicas}")
    print(f"  Temperature:    {args.temperature}")
    print(f"  Delta_E:        {args.delta_e}")
    print(f"  Seed:           {args.seed}")
    print("=" * 78)
    print()

    methods = ["random", "standard_mcmc", "parity_glauber", "sqa_uniform", "sqa_pdqm"]

    for target_bits in bit_sizes:
        curve = make_curve(target_bits)
        G = curve.generator
        n_bits = curve.key_bit_length()
        n_sweeps = max(args.base_sweeps, args.base_sweeps * n_bits // 8)

        print(f"--- {target_bits}-bit target -> {n_bits}-bit curve (p={str(curve.p)[:30]}) ---")
        print(f"    sweeps={n_sweeps}, sqa_steps={args.sqa_steps}, P={args.replicas}")

        config = ParityConfig(
            n_spins=n_bits,
            delta_e=args.delta_e,
            j_coupling=0.1,
            t1_base=0.05,
            t2=1.0,
            temperature=args.temperature,
            mode="exact" if n_bits <= 20 else "ising",
        )

        method_rates: dict[str, list[float]] = {m: [] for m in methods}

        for trial in range(args.n_trials):
            k, P = curve.random_keypair(rng)
            h = ParityHamiltonian.from_ec_dlp(curve, G, P, config)

            # 1. Random
            t0 = time.time()
            ek, mr = run_random(k, n_bits, rng)
            dt = time.time() - t0
            method_rates["random"].append(mr)
            all_results.append(TrialResult(
                n_bits=n_bits, method="random", true_key=k, extracted_key=ek,
                bit_match_rate=mr, key_recovered=(ek == k), energy_final=0.0,
                elapsed_sec=dt, trial_idx=trial, sweeps_or_steps=0,
            ))

            # 2. Standard MCMC
            t0 = time.time()
            ek, mr, e = run_standard_mcmc(h, k, n_sweeps, args.temperature, rng)
            dt = time.time() - t0
            method_rates["standard_mcmc"].append(mr)
            all_results.append(TrialResult(
                n_bits=n_bits, method="standard_mcmc", true_key=k, extracted_key=ek,
                bit_match_rate=mr, key_recovered=(ek == k), energy_final=e,
                elapsed_sec=dt, trial_idx=trial, sweeps_or_steps=n_sweeps,
            ))

            # 3. Parity Glauber
            t0 = time.time()
            ek, mr, e = run_parity_glauber(h, config, k, n_sweeps, args.temperature, rng)
            dt = time.time() - t0
            method_rates["parity_glauber"].append(mr)
            all_results.append(TrialResult(
                n_bits=n_bits, method="parity_glauber", true_key=k, extracted_key=ek,
                bit_match_rate=mr, key_recovered=(ek == k), energy_final=e,
                elapsed_sec=dt, trial_idx=trial, sweeps_or_steps=n_sweeps,
            ))

            # 4. SQA uniform (no parity weighting)
            t0 = time.time()
            ek, mr, e, agr = run_sqa(
                h, config, k, args.sqa_steps, args.replicas,
                parity_weighted=False, rng=rng,
            )
            dt = time.time() - t0
            method_rates["sqa_uniform"].append(mr)
            all_results.append(TrialResult(
                n_bits=n_bits, method="sqa_uniform", true_key=k, extracted_key=ek,
                bit_match_rate=mr, key_recovered=(ek == k), energy_final=e,
                elapsed_sec=dt, trial_idx=trial, sweeps_or_steps=args.sqa_steps,
            ))

            # 5. SQA PDQM (parity-weighted J_perp)
            t0 = time.time()
            ek, mr, e, agr = run_sqa(
                h, config, k, args.sqa_steps, args.replicas,
                parity_weighted=True, rng=rng,
            )
            dt = time.time() - t0
            method_rates["sqa_pdqm"].append(mr)
            all_results.append(TrialResult(
                n_bits=n_bits, method="sqa_pdqm", true_key=k, extracted_key=ek,
                bit_match_rate=mr, key_recovered=(ek == k), energy_final=e,
                elapsed_sec=dt, trial_idx=trial, sweeps_or_steps=args.sqa_steps,
            ))

            sys.stdout.write(f"\r    trial {trial+1}/{args.n_trials}")
            sys.stdout.flush()

        print()
        print(f"    {'Method':<20} {'Mean Match':>12} {'Std':>8} {'Keys':>8} {'Avg Time':>10}")
        print(f"    {'-'*58}")
        for method in methods:
            rates = method_rates[method]
            arr = np.array(rates)
            n_found = sum(
                1 for r in all_results
                if r.n_bits == n_bits and r.method == method and r.key_recovered
            )
            times = [
                r.elapsed_sec for r in all_results
                if r.n_bits == n_bits and r.method == method
            ]
            avg_time = np.mean(times) if times else 0
            print(f"    {method:<20} {arr.mean():>12.4f} {arr.std():>8.4f} {n_found:>5}/{len(rates)} {avg_time:>9.3f}s")
        print()

    # === Grand Summary ===
    print("=" * 78)
    print(" GRAND SUMMARY -- SQA vs CLASSICAL")
    print("=" * 78)
    print()

    for method in methods:
        results_m = [r for r in all_results if r.method == method]
        if not results_m:
            continue
        rates = [r.bit_match_rate for r in results_m]
        n_found = sum(1 for r in results_m if r.key_recovered)
        arr = np.array(rates)
        print(f"  {method:<20}  mean={arr.mean():.4f}  std={arr.std():.4f}  keys={n_found}/{len(rates)}")

    # Scaling table
    print()
    header = f"  {'Bits':>6}"
    for m in methods:
        header += f"  {m[:10]:>10}"
    print(header)
    print(f"  {'-'*66}")
    for target_bits in bit_sizes:
        n_bits = make_curve(target_bits).key_bit_length()
        row = f"  {target_bits:>6}"
        for method in methods:
            rates = [r.bit_match_rate for r in all_results
                     if r.method == method and r.n_bits == n_bits]
            if rates:
                row += f"  {np.mean(rates):>10.4f}"
            else:
                row += f"  {'---':>10}"
        print(row)

    # Statistical tests: SQA PDQM vs each baseline
    print()
    sqa_pdqm = np.array([r.bit_match_rate for r in all_results if r.method == "sqa_pdqm"])
    for baseline in ["random", "standard_mcmc", "parity_glauber", "sqa_uniform"]:
        bl_rates = np.array([r.bit_match_rate for r in all_results if r.method == baseline])
        if len(sqa_pdqm) > 0 and len(bl_rates) > 0:
            diff = sqa_pdqm.mean() - bl_rates.mean()
            pooled = np.sqrt(sqa_pdqm.var() / len(sqa_pdqm) + bl_rates.var() / len(bl_rates))
            z = diff / pooled if pooled > 0 else 0
            verdict = ""
            if z > 2.58:
                verdict = "REJECT null (p < 0.01)"
            elif z > 1.96:
                verdict = "REJECT null (p < 0.05)"
            elif z > 1.64:
                verdict = "MARGINAL (p < 0.10)"
            else:
                verdict = "FAIL TO REJECT null"
            print(f"  SQA_PDQM vs {baseline:<16}: diff={diff:+.4f}  Z={z:.2f}  -> {verdict}")

    print("=" * 78)

    # Export CSV
    desktop = os.path.expanduser("~/Desktop")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(desktop, f"sqa_ecdlp_{timestamp}.csv")

    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "n_bits", "method", "trial", "bit_match_rate", "key_recovered",
            "energy_final", "elapsed_sec", "sweeps_or_steps",
        ])
        for r in all_results:
            writer.writerow([
                r.n_bits, r.method, r.trial_idx, f"{r.bit_match_rate:.4f}",
                r.key_recovered, f"{r.energy_final:.4f}", f"{r.elapsed_sec:.4f}",
                r.sweeps_or_steps,
            ])

    print(f"\nResults exported to {filepath}")


if __name__ == "__main__":
    main()
