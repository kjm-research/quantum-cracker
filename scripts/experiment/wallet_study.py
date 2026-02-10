"""Wallet Correlation Study.

Generate 10 wallets with known private keys, run each private key through
the harmonic profiler, then correlate address features with harmonic features.

Output:
  ~/Desktop/wallet_correlation_report.txt
  ~/Desktop/wallet_correlation_matrix.csv
"""

import csv
import json
import os
import sys
import time

import numpy as np
from scipy.stats import pearsonr

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")

from experiment.crypto_utils import generate_wallet, address_features
from experiment.profiler import profile_single_key

NUM_WALLETS = 10


def extract_harmonic_features(profile):
    """Flatten harmonic profile into a feature vector for correlation."""
    features = {}

    # SH readback
    features["sh_match_rate"] = profile["sh_readback"]["match_rate"]
    sh_mags = profile["sh_readback"]["magnitudes"]
    features["sh_mean_magnitude"] = float(np.mean(sh_mags))
    features["sh_std_magnitude"] = float(np.std(sh_mags))
    features["sh_max_magnitude"] = float(np.max(sh_mags))

    # Count of 1-bits in SH extraction
    features["sh_ones_count"] = sum(profile["sh_readback"]["bits"])

    # LSQ
    features["lsq_match_rate"] = profile["lsq_inversion"]["match_rate"]
    features["lsq_condition_number_log"] = float(np.log10(profile["lsq_inversion"]["condition_number"]))
    features["lsq_residual"] = profile["lsq_inversion"]["residual_norm"]

    # Eigenvalues
    eigs = profile["eigenvalues"]["top_20"]
    features["ground_state_energy"] = eigs[0]
    features["eigenvalue_spread"] = eigs[-1] - eigs[0]
    features["eigenvalue_mean"] = float(np.mean(eigs))

    # Energy
    features["total_energy"] = profile["energy_stats"]["total_energy"]
    features["mean_energy"] = profile["energy_stats"]["mean_energy"]
    features["max_energy"] = profile["energy_stats"]["max_energy"]

    # Peaks
    if profile["peaks"]:
        peak_energies = [p["energy"] for p in profile["peaks"]]
        features["peak_mean_energy"] = float(np.mean(peak_energies))
        features["peak_max_energy"] = float(np.max(peak_energies))
        features["num_peaks"] = len(profile["peaks"])
    else:
        features["peak_mean_energy"] = 0.0
        features["peak_max_energy"] = 0.0
        features["num_peaks"] = 0

    # Key bit statistics
    bits = profile["key_bits"]
    features["key_ones_count"] = sum(bits)
    features["key_leading_zeros"] = next((i for i, b in enumerate(bits) if b == 1), 256)

    return features


def extract_address_features_flat(wallet):
    """Flatten wallet address features into a feature vector."""
    features = {}
    ef = wallet["eth_features"]

    features["eth_entropy"] = ef["entropy"]
    features["eth_longest_run"] = ef["longest_run"]
    features["eth_leading_zeros"] = ef["leading_zeros"]
    features["eth_trailing_zeros"] = ef["trailing_zeros"]
    features["eth_digit_sum"] = ef["digit_sum"]
    features["eth_unique_chars"] = ef["unique_chars"]

    # Nibble frequencies
    for i, count in enumerate(ef["nibble_freq"]):
        features[f"eth_nibble_{i:x}"] = count

    # Public key features
    pub_x = int(wallet["public_key_x"], 16)
    pub_y = int(wallet["public_key_y"], 16)
    features["pub_x_leading_zeros"] = len(wallet["public_key_x"]) - len(wallet["public_key_x"].lstrip("0"))
    features["pub_y_leading_zeros"] = len(wallet["public_key_y"]) - len(wallet["public_key_y"].lstrip("0"))
    features["pub_x_ones_count"] = bin(pub_x).count("1")
    features["pub_y_ones_count"] = bin(pub_y).count("1")

    return features


def compute_correlations(harmonic_features_list, address_features_list):
    """Compute Pearson correlation between all harmonic and address feature pairs."""
    h_keys = sorted(harmonic_features_list[0].keys())
    a_keys = sorted(address_features_list[0].keys())

    results = []
    for hk in h_keys:
        h_values = [hf[hk] for hf in harmonic_features_list]
        for ak in a_keys:
            a_values = [af[ak] for af in address_features_list]

            # Skip if constant (no variance)
            if np.std(h_values) < 1e-10 or np.std(a_values) < 1e-10:
                continue

            try:
                corr, p_val = pearsonr(h_values, a_values)
                results.append({
                    "harmonic_feature": hk,
                    "address_feature": ak,
                    "correlation": corr,
                    "p_value": p_val,
                    "abs_correlation": abs(corr),
                })
            except Exception:
                continue

    return results


def main():
    print("=" * 70)
    print("  WALLET CORRELATION STUDY")
    print("=" * 70)

    t0 = time.time()

    # Generate 10 wallets
    print(f"\nGenerating {NUM_WALLETS} wallets...")
    wallets = []
    for i in range(NUM_WALLETS):
        w = generate_wallet()
        wallets.append(w)
        print(f"  [{i+1}] ETH: {w['eth_address']}")
        print(f"       BTC: {w['btc_address']}")
        print(f"       Key: {w['private_key_hex'][:16]}...")

    # Profile each private key
    print(f"\nProfiling {NUM_WALLETS} private keys through harmonic system...")
    profiles = []
    for i, w in enumerate(wallets):
        profile = profile_single_key(w["private_key_hex"], i)
        profiles.append(profile)

    # Extract feature vectors
    harmonic_features_list = [extract_harmonic_features(p) for p in profiles]
    address_features_list = [extract_address_features_flat(w) for w in wallets]

    # Compute correlations
    print(f"\nComputing correlations...")
    correlations = compute_correlations(harmonic_features_list, address_features_list)

    # Sort by absolute correlation
    correlations.sort(key=lambda x: x["abs_correlation"], reverse=True)

    # Report
    report_lines = []
    report_lines.append("WALLET CORRELATION STUDY REPORT")
    report_lines.append(f"Wallets analyzed: {NUM_WALLETS}")
    report_lines.append(f"Harmonic features: {len(harmonic_features_list[0])}")
    report_lines.append(f"Address features: {len(address_features_list[0])}")
    report_lines.append(f"Total correlations computed: {len(correlations)}")
    report_lines.append("")

    # Bonferroni correction
    n_tests = len(correlations)
    bonferroni_threshold = 0.05 / n_tests if n_tests > 0 else 0.05
    significant = [c for c in correlations if c["p_value"] < bonferroni_threshold]

    report_lines.append(f"Bonferroni-corrected significance threshold: p < {bonferroni_threshold:.6f}")
    report_lines.append(f"Significant correlations: {len(significant)}")
    report_lines.append("")

    report_lines.append("=" * 70)
    report_lines.append("TOP 30 CORRELATIONS (by |r|)")
    report_lines.append("=" * 70)

    for c in correlations[:30]:
        sig = "***" if c["p_value"] < bonferroni_threshold else ""
        report_lines.append(
            f"  r={c['correlation']:+.4f}  p={c['p_value']:.6f}  "
            f"{c['harmonic_feature']:>30s} vs {c['address_feature']:<25s} {sig}"
        )

    if significant:
        report_lines.append("")
        report_lines.append("=" * 70)
        report_lines.append("SIGNIFICANT CORRELATIONS (Bonferroni-corrected)")
        report_lines.append("=" * 70)
        for c in significant:
            report_lines.append(
                f"  r={c['correlation']:+.4f}  p={c['p_value']:.6f}  "
                f"{c['harmonic_feature']} vs {c['address_feature']}"
            )
    else:
        report_lines.append("")
        report_lines.append("NO SIGNIFICANT CORRELATIONS FOUND (as expected for random keys)")

    # Wallet details
    report_lines.append("")
    report_lines.append("=" * 70)
    report_lines.append("WALLET DETAILS")
    report_lines.append("=" * 70)
    for i, w in enumerate(wallets):
        report_lines.append(f"\n  Wallet {i+1}:")
        report_lines.append(f"    Private key: {w['private_key_hex']}")
        report_lines.append(f"    ETH address: {w['eth_address']}")
        report_lines.append(f"    BTC address: {w['btc_address']}")
        report_lines.append(f"    SH match:    {profiles[i]['sh_readback']['match_rate']:.4f}")

    total_time = time.time() - t0
    report_lines.append(f"\nRuntime: {total_time:.1f}s")

    report = "\n".join(report_lines)
    print(report)

    # Save report
    report_path = os.path.expanduser("~/Desktop/wallet_correlation_report.txt")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"\nReport saved: {report_path}")

    # Save correlation matrix as CSV
    csv_path = os.path.expanduser("~/Desktop/wallet_correlation_matrix.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["harmonic_feature", "address_feature", "correlation", "p_value"])
        writer.writeheader()
        for c in correlations:
            writer.writerow({
                "harmonic_feature": c["harmonic_feature"],
                "address_feature": c["address_feature"],
                "correlation": f"{c['correlation']:.6f}",
                "p_value": f"{c['p_value']:.6f}",
            })
    print(f"Correlation matrix saved: {csv_path}")


if __name__ == "__main__":
    main()
