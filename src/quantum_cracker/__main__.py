"""Main entry point: python -m quantum_cracker"""

from __future__ import annotations

import argparse
import csv
import os
from datetime import datetime

import numpy as np

from quantum_cracker import __version__
from quantum_cracker.core.harmonic_compiler import HarmonicCompiler
from quantum_cracker.core.key_interface import KeyInput
from quantum_cracker.core.rip_engine import RipEngine
from quantum_cracker.core.voxel_grid import SphericalVoxelGrid
from quantum_cracker.analysis.metrics import MetricExtractor
from quantum_cracker.analysis.validation import Validator
from quantum_cracker.utils.types import SimulationConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="quantum-cracker",
        description="Harmonic Spherical Compiler -- resolve 256-bit keys via 78 MHz resonance",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    sub = parser.add_subparsers(dest="command")

    # simulate
    sim = sub.add_parser("simulate", help="Run full simulation pipeline")
    sim.add_argument("--key", type=str, help="256-bit key (hex, binary, or int)")
    sim.add_argument("--random", action="store_true", help="Generate random key")
    sim.add_argument("--steps", type=int, default=100, help="Simulation timesteps")
    sim.add_argument("--grid-size", type=int, default=20, help="Voxel grid size (default 20)")
    sim.add_argument("--no-viz", action="store_true", help="Skip plot generation")
    sim.add_argument("--csv", action="store_true", help="Export results CSV to ~/Desktop")
    sim.add_argument("--sh-filter", action="store_true", help="Apply SH filter during compilation")

    # parity
    par = sub.add_parser("parity", help="Parity-driven ECDLP via Ising Hamiltonian")
    par.add_argument("--curve-bits", type=int, default=8, help="EC curve size in bits (default 8)")
    par.add_argument("--trajectories", type=int, default=100, help="Annealing trajectories")
    par.add_argument("--anneal-steps", type=int, default=500, help="Steps per anneal trajectory")
    par.add_argument("--beta-final", type=float, default=20.0, help="Final inverse temperature")
    par.add_argument("--delta-e", type=float, default=2.0, help="Parity energy gap")
    par.add_argument("--csv", action="store_true", help="Export results CSV to ~/Desktop")
    par.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")

    # visualize
    viz = sub.add_parser("visualize", help="Launch 3D renderer with live resonance")
    viz.add_argument("--key", type=str, help="256-bit key (hex)")
    viz.add_argument("--random", action="store_true", help="Generate random key")
    viz.add_argument("--grid-size", type=int, default=78, help="Voxel grid size (default 78)")

    return parser


def get_key(args: argparse.Namespace) -> KeyInput:
    """Resolve key from args."""
    if getattr(args, "random", False):
        key = KeyInput.random()
        print(f"Generated random key: {key.as_hex}")
        return key
    elif getattr(args, "key", None):
        return KeyInput(args.key)
    else:
        return KeyInput.from_cli()


def run_simulation(args: argparse.Namespace) -> None:
    """Full pipeline: key -> engines -> simulate -> analyze -> report."""
    key = get_key(args)
    grid_size = args.grid_size
    steps = args.steps

    print(f"Key: {key.as_hex}")
    print(f"Grid: {grid_size}x{grid_size}x{grid_size} | Steps: {steps}")
    print()

    # Initialize
    config = SimulationConfig(
        grid_size=grid_size,
        timesteps=steps,
    )

    print("Initializing voxel grid...")
    grid = SphericalVoxelGrid(size=grid_size)
    grid.initialize_from_key(key)

    print("Initializing rip engine...")
    engine = RipEngine(config=config)
    engine.initialize_from_key(key)

    # Run rip engine
    print(f"Running rip engine ({steps} steps)...")
    rip_history = engine.run(steps)

    # Run harmonic compiler
    print(f"Running harmonic compiler ({steps} steps)...")
    compiler = HarmonicCompiler(grid, config=config)
    peaks = compiler.compile(
        num_steps=steps,
        dt=0.01,
        apply_sh_filter=args.sh_filter,
    )

    # Hamiltonian
    print("Computing Hamiltonian eigenvalues...")
    eigenvalues = compiler.compute_hamiltonian_eigenvalues()

    # Analysis
    print("Extracting metrics...")
    extractor = MetricExtractor(peaks, rip_history)
    report = extractor.full_report()
    extracted_bits = extractor.peaks_to_key_bits()

    # Validation
    validator = Validator(key, extracted_bits)
    validation = validator.summary(
        total_peaks=len(peaks),
        peaks_theta=[p.theta for p in peaks],
    )

    # Output
    print()
    print("=" * 50)
    print(" RESULTS")
    print("=" * 50)
    print(f"  Peaks extracted:     {len(peaks)}")
    print(f"  Bit match rate:      {validation['bit_match_rate']:.4f}")
    print(f"  Confidence interval: ({validation['confidence_interval'][0]:.3f}, {validation['confidence_interval'][1]:.3f})")
    print(f"  Peak alignment:      {validation['peak_alignment']:.4f}")
    print(f"  Ghost harmonics:     {validation['ghost_count']}")
    print(f"  Ground state energy: {eigenvalues[0]:.4f}")
    if len(eigenvalues) > 1:
        print(f"  Energy gap:          {eigenvalues[1] - eigenvalues[0]:.4f}")
    print(f"  Visible threads:     {engine.num_visible}/{engine.num_threads}")
    print(f"  Final radius:        {engine.radius:.2e} m")
    print("=" * 50)

    # CSV export
    if args.csv:
        export_csv(key, report, validation, eigenvalues)

    # Plots
    if not args.no_viz:
        from quantum_cracker.visualization.plots import PlotSuite

        print("\nGenerating plots...")
        plots = PlotSuite()
        plots.spherical_harmonic_heatmap(grid)
        plots.thread_gap_vs_time(rip_history)
        plots.energy_landscape(eigenvalues)
        plots.key_comparison(key, extracted_bits)
        plots.peak_distribution_3d(peaks)
        print("Plots saved to ~/Desktop/")


def export_csv(
    key: KeyInput,
    report: dict,
    validation: dict,
    eigenvalues,
) -> None:
    """Write results to ~/Desktop/quantum_cracker_<timestamp>.csv"""
    desktop = os.path.expanduser("~/Desktop")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(desktop, f"quantum_cracker_{timestamp}.csv")

    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["key_hex", key.as_hex])
        writer.writerow(["bit_match_rate", validation["bit_match_rate"]])
        writer.writerow(["peak_alignment", validation["peak_alignment"]])
        writer.writerow(["confidence_lo", validation["confidence_interval"][0]])
        writer.writerow(["confidence_hi", validation["confidence_interval"][1]])
        writer.writerow(["ghost_count", validation["ghost_count"]])
        writer.writerow(["ground_state_energy", float(eigenvalues[0])])
        if len(eigenvalues) > 1:
            writer.writerow(["energy_gap", float(eigenvalues[1] - eigenvalues[0])])
        for k, v in report.get("peak_stats", {}).items():
            writer.writerow([f"peak_{k}", v])
        for k, v in report.get("thread_stats", {}).items():
            writer.writerow([f"thread_{k}", v])

    print(f"Results exported to {filepath}")


def run_visualize(args: argparse.Namespace) -> None:
    """Launch 3D renderer with live harmonic resonance."""
    key = get_key(args)
    grid_size = args.grid_size

    print(f"Key: {key.as_hex}")
    print(f"Grid: {grid_size}x{grid_size}x{grid_size}")
    print()

    print("Initializing voxel grid...")
    grid = SphericalVoxelGrid(size=grid_size)
    grid.initialize_from_key(key)

    print("Initializing rip engine...")
    engine = RipEngine()
    engine.initialize_from_key(key)

    print("Initializing harmonic compiler...")
    config = SimulationConfig(grid_size=grid_size)
    compiler = HarmonicCompiler(grid, config=config)

    print("Launching 3D renderer...")
    print("  SPACE=pause  R=reset  +/-=speed  1/2=toggle  ESC=close")
    print()

    from quantum_cracker.visualization.renderer import QuantumRenderer

    renderer = QuantumRenderer(grid, engine, compiler=compiler)
    renderer.run()


def run_parity(args: argparse.Namespace) -> None:
    """Parity-driven ECDLP: Ising Hamiltonian + parity dynamics."""
    from quantum_cracker.parity.ec_constraints import SmallEC
    from quantum_cracker.parity.hamiltonian import ParityHamiltonian
    from quantum_cracker.parity.oracle import ParityOracle
    from quantum_cracker.parity.types import AnnealSchedule, ParityConfig

    seed = args.seed if args.seed is not None else 42
    rng = np.random.default_rng(seed)

    # Pick a curve with approximately the right bit size
    curve_primes = [97, 251, 509, 1021, 2039, 4093, 8191, 16381, 32749, 65521]
    target_bits = args.curve_bits
    p = 97
    for cp in curve_primes:
        if cp.bit_length() >= target_bits:
            p = cp
            break
    else:
        p = curve_primes[-1]

    curve = SmallEC(p, 0, 7)
    G = curve.generator
    k, P = curve.random_keypair(rng)
    n_bits = curve.key_bit_length()

    print(f"Curve: y^2 = x^3 + 7 over F_{p}")
    print(f"Order: {curve.order} ({n_bits} bits)")
    print(f"Generator: {G}")
    print(f"Private key: {k}")
    print(f"Public key: {P}")
    print()

    config = ParityConfig(
        n_spins=n_bits,
        delta_e=args.delta_e,
        j_coupling=0.1,
        t1_base=0.05,
        t2=1.0,
        mode="exact",
        constraint_weight=20.0,
    )

    print("Building parity Hamiltonian...")
    h = ParityHamiltonian.from_ec_dlp(curve, G, P, config)

    # Verify ground state
    gs_key = h.ground_state_key()
    gs_point = curve.multiply(G, gs_key)
    print(f"Hamiltonian ground state key: {gs_key}")
    print(f"Ground state -> {gs_key}*G = {gs_point}")
    print(f"Matches public key: {gs_point == P}")
    print()

    # Run oracle
    print(f"Running parity oracle ({args.trajectories} trajectories, {args.anneal_steps} steps)...")
    oracle = ParityOracle(h, config)
    schedule = AnnealSchedule(
        n_steps=args.anneal_steps,
        beta_initial=0.1,
        beta_final=args.beta_final,
    )
    result = oracle.measure(
        n_trajectories=args.trajectories,
        schedule=schedule,
        target_key=k,
        rng=rng,
    )

    match_rate = oracle.bit_match_rate(result, k)
    extracted_key = oracle.extract_key(result)

    print()
    print("=" * 50)
    print(" PARITY ORACLE RESULTS")
    print("=" * 50)
    print(f"  True key:            {k}")
    print(f"  Extracted key:       {extracted_key}")
    print(f"  Bit match rate:      {match_rate:.4f}")
    print(f"  Mean confidence:     {float(result.bit_confidences.mean()):.4f}")
    print(f"  Best energy:         {result.best_energy:.4f}")
    print(f"  Mean energy:         {result.mean_energy:.4f}")
    print(f"  Parity distribution: even={result.parity_distribution.get(1, 0)}, odd={result.parity_distribution.get(-1, 0)}")
    print(f"  Key recovered:       {extracted_key == k}")
    print("=" * 50)

    if args.csv:
        desktop = os.path.expanduser("~/Desktop")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(desktop, f"parity_ecdlp_{timestamp}.csv")

        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "value"])
            writer.writerow(["curve_p", p])
            writer.writerow(["curve_order", curve.order])
            writer.writerow(["n_bits", n_bits])
            writer.writerow(["true_key", k])
            writer.writerow(["extracted_key", extracted_key])
            writer.writerow(["bit_match_rate", match_rate])
            writer.writerow(["mean_confidence", float(result.bit_confidences.mean())])
            writer.writerow(["best_energy", result.best_energy])
            writer.writerow(["mean_energy", result.mean_energy])
            writer.writerow(["parity_even", result.parity_distribution.get(1, 0)])
            writer.writerow(["parity_odd", result.parity_distribution.get(-1, 0)])
            writer.writerow(["key_recovered", extracted_key == k])
            writer.writerow(["n_trajectories", args.trajectories])
            writer.writerow(["anneal_steps", args.anneal_steps])
            writer.writerow(["delta_e", args.delta_e])
            writer.writerow(["beta_final", args.beta_final])

        print(f"Results exported to {filepath}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "simulate":
        run_simulation(args)
    elif args.command == "parity":
        run_parity(args)
    elif args.command == "visualize":
        run_visualize(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
