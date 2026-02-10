"""Run the full experiment pipeline.

1. Profile 10 random keys (profiler.py)
2. Analyze cross-key patterns (profile_analysis.py)
3. Wallet correlation study (wallet_study.py)

Usage:
    python scripts/experiment/run_experiment.py [--profile-only] [--wallet-only] [--analysis-only]
"""

import os
import sys
import time

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")


def run_profiler():
    print("\n" + "=" * 70)
    print("  PHASE 1: HARMONIC PROFILING (10 keys)")
    print("=" * 70)
    from experiment.profiler import main as profiler_main
    profiler_main()


def run_analysis():
    print("\n" + "=" * 70)
    print("  PHASE 1b: CROSS-KEY ANALYSIS")
    print("=" * 70)
    profiles_path = os.path.expanduser("~/Desktop/harmonic_profiles.json")
    if not os.path.exists(profiles_path):
        print(f"  ERROR: {profiles_path} not found. Run profiler first.")
        return
    from experiment.profile_analysis import main as analysis_main
    analysis_main()


def run_wallet_study():
    print("\n" + "=" * 70)
    print("  PHASE 2: WALLET CORRELATION STUDY")
    print("=" * 70)
    from experiment.wallet_study import main as wallet_main
    wallet_main()


def main():
    args = sys.argv[1:]

    t0 = time.time()

    if "--profile-only" in args:
        run_profiler()
    elif "--analysis-only" in args:
        run_analysis()
    elif "--wallet-only" in args:
        run_wallet_study()
    else:
        # Full pipeline
        run_profiler()
        run_analysis()
        run_wallet_study()

    total = time.time() - t0
    print(f"\n{'='*70}")
    print(f"  EXPERIMENT COMPLETE -- {total:.1f}s total")
    print(f"{'='*70}")
    print(f"  Reports on ~/Desktop:")
    print(f"    harmonic_profiles.json")
    print(f"    profile_analysis_report.txt")
    print(f"    wallet_correlation_report.txt")
    print(f"    wallet_correlation_matrix.csv")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
