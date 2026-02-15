"""Parity-Driven ECDLP Experiment.

Tests whether PDQM parity dynamics provide any advantage over standard
MCMC for the elliptic curve discrete logarithm problem.

The experiment runs three comparisons on toy EC curves of increasing size:
  1. Random guessing (baseline, ~50% bit match rate)
  2. Standard Metropolis MCMC (single-spin flips, no parity weighting)
  3. PDQM parity dynamics (pair hopping + parity suppression + coherence boost)

Null hypothesis: "Parity dynamics provides no advantage over standard MCMC for ECDLP."
The experiment is designed to honestly reject or fail to reject this hypothesis.

Output: CSV to ~/Desktop with all results.

Usage:
    python scripts/experiment/parity_ecdlp.py
    python scripts/experiment/parity_ecdlp.py --curves 97,251,509
    python scripts/experiment/parity_ecdlp.py --n-trials 20 --trajectories 200
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

sys.path.insert(0, "src")

from quantum_cracker.parity.dynamics import ParityDynamics, compute_parity
from quantum_cracker.parity.ec_constraints import SmallEC
from quantum_cracker.parity.hamiltonian import ParityHamiltonian
from quantum_cracker.parity.oracle import ParityOracle
from quantum_cracker.parity.types import AnnealSchedule, ParityConfig


@dataclass
class TrialResult:
    """Result from a single trial (one keypair, one method)."""
    curve_p: int
    n_bits: int
    method: str
    true_key: int
    extracted_key: int
    bit_match_rate: float
    key_recovered: bool
    energy_final: float
    parity_even_frac: float
    elapsed_sec: float
    trial_idx: int


def run_random_baseline(
    true_key: int, n_bits: int, rng: np.random.Generator
) -> tuple[int, float]:
    """Random guessing baseline."""
    guess = int(rng.integers(0, 1 << n_bits))
    xor = guess ^ true_key
    matching = n_bits - bin(xor).count("1")
    return guess, matching / n_bits


def run_standard_mcmc(
    hamiltonian: ParityHamiltonian,
    true_key: int,
    n_sweeps: int,
    temperature: float,
    rng: np.random.Generator,
) -> tuple[int, float, float]:
    """Standard Metropolis MCMC (no parity weighting)."""
    n = hamiltonian.n_spins
    sigma0 = rng.choice([-1, 1], size=n).astype(np.int8)

    config = hamiltonian.config
    dyn = ParityDynamics(hamiltonian, config)
    snaps = dyn.evolve_standard_mcmc(
        sigma0, n_sweeps=n_sweeps, temperature=temperature,
        target_key=true_key, log_interval=n_sweeps,
        rng=rng,
    )

    final_spins = snaps[-1].spins
    extracted = ParityHamiltonian.spins_to_key(final_spins)
    xor = extracted ^ true_key
    matching = n - bin(xor).count("1")
    return extracted, matching / n, snaps[-1].energy


def run_parity_dynamics(
    hamiltonian: ParityHamiltonian,
    config: ParityConfig,
    true_key: int,
    n_sweeps: int,
    temperature: float,
    rng: np.random.Generator,
) -> tuple[int, float, float]:
    """PDQM parity dynamics (pair hopping + suppression)."""
    n = hamiltonian.n_spins
    sigma0 = rng.choice([-1, 1], size=n).astype(np.int8)

    dyn = ParityDynamics(hamiltonian, config)
    snaps = dyn.evolve_glauber(
        sigma0, n_sweeps=n_sweeps, temperature=temperature,
        target_key=true_key, log_interval=n_sweeps,
        rng=rng,
    )

    final_spins = snaps[-1].spins
    extracted = ParityHamiltonian.spins_to_key(final_spins)
    xor = extracted ^ true_key
    matching = n - bin(xor).count("1")
    return extracted, matching / n, snaps[-1].energy


def run_parity_oracle(
    hamiltonian: ParityHamiltonian,
    config: ParityConfig,
    true_key: int,
    n_trajectories: int,
    anneal_steps: int,
    rng: np.random.Generator,
) -> tuple[int, float, float, float]:
    """Full parity oracle (annealing + parity-weighted voting)."""
    oracle = ParityOracle(hamiltonian, config)
    schedule = AnnealSchedule(
        n_steps=anneal_steps,
        beta_initial=0.1,
        beta_final=20.0,
    )
    result = oracle.measure(
        n_trajectories=n_trajectories,
        schedule=schedule,
        target_key=true_key,
        rng=rng,
    )
    match_rate = oracle.bit_match_rate(result, true_key)
    extracted = oracle.extract_key(result)
    even_frac = result.parity_distribution.get(1, 0) / max(n_trajectories, 1)
    return extracted, match_rate, result.best_energy, even_frac


def main():
    parser = argparse.ArgumentParser(description="Parity-Driven ECDLP Experiment")
    parser.add_argument("--curves", type=str, default="97,251,509,1021",
                        help="Comma-separated curve primes (default: 97,251,509,1021)")
    parser.add_argument("--n-trials", type=int, default=10,
                        help="Trials per curve per method (default: 10)")
    parser.add_argument("--trajectories", type=int, default=100,
                        help="Annealing trajectories for oracle (default: 100)")
    parser.add_argument("--anneal-steps", type=int, default=500,
                        help="Steps per annealing trajectory (default: 500)")
    parser.add_argument("--mcmc-sweeps", type=int, default=200,
                        help="MCMC sweeps for Glauber/standard (default: 200)")
    parser.add_argument("--temperature", type=float, default=0.2,
                        help="MCMC temperature (default: 0.2)")
    parser.add_argument("--delta-e", type=float, default=2.0,
                        help="Parity energy gap (default: 2.0)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    args = parser.parse_args()

    curve_primes = [int(p.strip()) for p in args.curves.split(",")]
    rng = np.random.default_rng(args.seed)
    all_results: list[TrialResult] = []

    print("=" * 70)
    print(" PARITY-DRIVEN ECDLP EXPERIMENT")
    print("=" * 70)
    print(f"Curves: {curve_primes}")
    print(f"Trials per method: {args.n_trials}")
    print(f"Oracle trajectories: {args.trajectories}")
    print(f"Anneal steps: {args.anneal_steps}")
    print(f"MCMC sweeps: {args.mcmc_sweeps}")
    print(f"Temperature: {args.temperature}")
    print(f"Delta_E: {args.delta_e}")
    print(f"Seed: {args.seed}")
    print("=" * 70)
    print()

    for p in curve_primes:
        curve = SmallEC(p, 0, 7)
        G = curve.generator
        n_bits = curve.key_bit_length()

        print(f"--- Curve p={p}, order={curve.order}, bits={n_bits} ---")

        config = ParityConfig(
            n_spins=n_bits,
            delta_e=args.delta_e,
            j_coupling=0.1,
            t1_base=0.05,
            t2=1.0,
            temperature=args.temperature,
            mode="exact",
            constraint_weight=20.0,
        )

        method_totals = {
            "random": [],
            "standard_mcmc": [],
            "parity_glauber": [],
            "parity_oracle": [],
        }

        for trial in range(args.n_trials):
            k, P = curve.random_keypair(rng)
            h = ParityHamiltonian.from_ec_dlp(curve, G, P, config)

            # 1. Random baseline
            t0 = time.time()
            ext_r, mr_r = run_random_baseline(k, n_bits, rng)
            dt_r = time.time() - t0
            method_totals["random"].append(mr_r)
            all_results.append(TrialResult(
                curve_p=p, n_bits=n_bits, method="random",
                true_key=k, extracted_key=ext_r, bit_match_rate=mr_r,
                key_recovered=(ext_r == k), energy_final=0.0,
                parity_even_frac=0.5, elapsed_sec=dt_r, trial_idx=trial,
            ))

            # 2. Standard MCMC
            t0 = time.time()
            ext_s, mr_s, e_s = run_standard_mcmc(
                h, k, args.mcmc_sweeps, args.temperature, rng
            )
            dt_s = time.time() - t0
            method_totals["standard_mcmc"].append(mr_s)
            all_results.append(TrialResult(
                curve_p=p, n_bits=n_bits, method="standard_mcmc",
                true_key=k, extracted_key=ext_s, bit_match_rate=mr_s,
                key_recovered=(ext_s == k), energy_final=e_s,
                parity_even_frac=0.5, elapsed_sec=dt_s, trial_idx=trial,
            ))

            # 3. Parity Glauber
            t0 = time.time()
            ext_pg, mr_pg, e_pg = run_parity_dynamics(
                h, config, k, args.mcmc_sweeps, args.temperature, rng
            )
            dt_pg = time.time() - t0
            method_totals["parity_glauber"].append(mr_pg)
            all_results.append(TrialResult(
                curve_p=p, n_bits=n_bits, method="parity_glauber",
                true_key=k, extracted_key=ext_pg, bit_match_rate=mr_pg,
                key_recovered=(ext_pg == k), energy_final=e_pg,
                parity_even_frac=0.5, elapsed_sec=dt_pg, trial_idx=trial,
            ))

            # 4. Parity Oracle (annealing + voting)
            t0 = time.time()
            ext_po, mr_po, e_po, ef_po = run_parity_oracle(
                h, config, k, args.trajectories, args.anneal_steps, rng
            )
            dt_po = time.time() - t0
            method_totals["parity_oracle"].append(mr_po)
            all_results.append(TrialResult(
                curve_p=p, n_bits=n_bits, method="parity_oracle",
                true_key=k, extracted_key=ext_po, bit_match_rate=mr_po,
                key_recovered=(ext_po == k), energy_final=e_po,
                parity_even_frac=ef_po, elapsed_sec=dt_po, trial_idx=trial,
            ))

        # Print summary for this curve
        print(f"  {'Method':<20} {'Mean Match':>12} {'Std':>8} {'Keys Found':>12}")
        print(f"  {'-'*52}")
        for method, rates in method_totals.items():
            arr = np.array(rates)
            n_found = sum(
                1 for r in all_results
                if r.curve_p == p and r.method == method and r.key_recovered
            )
            print(f"  {method:<20} {arr.mean():>12.4f} {arr.std():>8.4f} {n_found:>8}/{args.n_trials}")
        print()

    # --- Grand summary ---
    print("=" * 70)
    print(" GRAND SUMMARY")
    print("=" * 70)

    for method in ["random", "standard_mcmc", "parity_glauber", "parity_oracle"]:
        rates = [r.bit_match_rate for r in all_results if r.method == method]
        n_found = sum(1 for r in all_results if r.method == method and r.key_recovered)
        n_total = len(rates)
        arr = np.array(rates)
        print(f"  {method:<20}  mean={arr.mean():.4f}  std={arr.std():.4f}  "
              f"keys={n_found}/{n_total}")

    # Statistical test: is parity oracle significantly better than standard MCMC?
    oracle_rates = np.array([r.bit_match_rate for r in all_results if r.method == "parity_oracle"])
    mcmc_rates = np.array([r.bit_match_rate for r in all_results if r.method == "standard_mcmc"])

    if len(oracle_rates) > 0 and len(mcmc_rates) > 0:
        diff = oracle_rates.mean() - mcmc_rates.mean()
        pooled_std = np.sqrt(oracle_rates.var() / len(oracle_rates) + mcmc_rates.var() / len(mcmc_rates))
        z_score = diff / pooled_std if pooled_std > 0 else 0
        print()
        print(f"  Parity oracle vs standard MCMC:")
        print(f"    Mean difference:  {diff:+.4f}")
        print(f"    Z-score:          {z_score:.2f}")
        if z_score > 1.96:
            print(f"    Result:           REJECT null (p < 0.05, parity dynamics helps)")
        elif z_score > 1.64:
            print(f"    Result:           MARGINAL (p < 0.10)")
        else:
            print(f"    Result:           FAIL TO REJECT null (no significant advantage)")
    print("=" * 70)

    # --- Export CSV ---
    desktop = os.path.expanduser("~/Desktop")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(desktop, f"parity_ecdlp_experiment_{timestamp}.csv")

    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "curve_p", "n_bits", "method", "trial", "true_key", "extracted_key",
            "bit_match_rate", "key_recovered", "energy_final",
            "parity_even_frac", "elapsed_sec",
        ])
        for r in all_results:
            writer.writerow([
                r.curve_p, r.n_bits, r.method, r.trial_idx, r.true_key,
                r.extracted_key, f"{r.bit_match_rate:.4f}", r.key_recovered,
                f"{r.energy_final:.4f}", f"{r.parity_even_frac:.4f}",
                f"{r.elapsed_sec:.4f}",
            ])

    print(f"\nResults exported to {filepath}")


if __name__ == "__main__":
    main()
