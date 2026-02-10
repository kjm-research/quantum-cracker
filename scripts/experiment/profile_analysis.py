"""Cross-key pattern analysis.

Loads harmonic_profiles.json (from profiler.py) and runs statistical
analysis looking for patterns across keys.

Output: ~/Desktop/profile_analysis_report.txt
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, "src")


def load_profiles(path=None):
    if path is None:
        path = os.path.expanduser("~/Desktop/harmonic_profiles.json")
    with open(path) as f:
        return json.load(f)


def analyze_coefficient_stability(profiles):
    """Which SH modes always have high/low confidence regardless of key?"""
    lines = []
    lines.append("=" * 70)
    lines.append("1. SH COEFFICIENT STABILITY ACROSS KEYS")
    lines.append("=" * 70)

    all_magnitudes = []
    for p in profiles:
        mags = np.array(p["sh_readback"]["magnitudes"])
        all_magnitudes.append(mags)

    mag_matrix = np.array(all_magnitudes)  # (num_keys, 256)
    mean_mag = np.mean(mag_matrix, axis=0)
    std_mag = np.std(mag_matrix, axis=0)
    cv = std_mag / (mean_mag + 1e-30)  # coefficient of variation

    # Most stable (low CV) = same magnitude regardless of key
    stable_idx = np.argsort(cv)[:20]
    lines.append(f"\n  Top 20 most stable SH modes (low coefficient of variation):")
    for idx in stable_idx:
        lines.append(f"    Bit {idx:3d}: mean_mag={mean_mag[idx]:.6f}, std={std_mag[idx]:.6f}, CV={cv[idx]:.4f}")

    # Most variable (high CV) = key-dependent
    variable_idx = np.argsort(cv)[::-1][:20]
    lines.append(f"\n  Top 20 most variable SH modes (key-dependent):")
    for idx in variable_idx:
        lines.append(f"    Bit {idx:3d}: mean_mag={mean_mag[idx]:.6f}, std={std_mag[idx]:.6f}, CV={cv[idx]:.4f}")

    # Bits that are always correct vs always wrong
    all_sh_matches = []
    for p in profiles:
        sh_bits = p["sh_readback"]["bits"]
        target = p["key_bits"]
        matches = [1 if sh_bits[i] == target[i] else 0 for i in range(256)]
        all_sh_matches.append(matches)

    match_matrix = np.array(all_sh_matches)  # (num_keys, 256)
    per_bit_success = np.mean(match_matrix, axis=0)

    always_correct = np.where(per_bit_success == 1.0)[0]
    always_wrong = np.where(per_bit_success == 0.0)[0]
    lines.append(f"\n  Bits ALWAYS correctly extracted (all keys): {len(always_correct)}")
    lines.append(f"  Bits NEVER correctly extracted (all keys):  {len(always_wrong)}")
    lines.append(f"  Bits sometimes correct:                     {256 - len(always_correct) - len(always_wrong)}")

    # Correlation between magnitude and correctness
    flat_mag = mag_matrix.ravel()
    flat_match = match_matrix.ravel()
    corr = np.corrcoef(flat_mag, flat_match)[0, 1]
    lines.append(f"\n  Correlation (|coefficient| vs correctness): {corr:.4f}")

    return "\n".join(lines), per_bit_success, mean_mag


def analyze_frequency_patterns(profiles):
    """Do certain frequencies always crack certain bit positions?"""
    lines = []
    lines.append("\n" + "=" * 70)
    lines.append("2. FREQUENCY RESONANCE PATTERNS")
    lines.append("=" * 70)

    # Collect per-bit match rates across all keys for each frequency
    all_freqs = set()
    for p in profiles:
        all_freqs.update(p["frequency_sweep"].keys())
    all_freqs = sorted(all_freqs, key=lambda x: int(x))

    # For each frequency, average match rate across keys
    freq_avg_rates = {}
    for freq in all_freqs:
        rates = []
        for p in profiles:
            if freq in p["frequency_sweep"]:
                rates.append(p["frequency_sweep"][freq]["match_rate"])
        freq_avg_rates[freq] = np.mean(rates) if rates else 0.0

    sorted_freqs = sorted(freq_avg_rates.keys(), key=lambda f: freq_avg_rates[f], reverse=True)
    lines.append(f"\n  Top 10 frequencies by average match rate across keys:")
    for f in sorted_freqs[:10]:
        lines.append(f"    {f:>4s} MHz: avg rate = {freq_avg_rates[f]:.4f}")

    # Per-bit: how many keys does each frequency crack this bit?
    freq_bit_consistency = {}
    for freq in all_freqs[:10]:  # top 10 freqs
        per_bit_count = np.zeros(256)
        for p in profiles:
            if freq in p["frequency_sweep"]:
                matches = p["frequency_sweep"][freq]["per_bit_match"]
                per_bit_count += np.array(matches)
        freq_bit_consistency[freq] = per_bit_count

    lines.append(f"\n  Frequency-bit consistency (top freq, bits correct in all keys):")
    for freq in list(freq_bit_consistency.keys())[:5]:
        counts = freq_bit_consistency[freq]
        all_keys = int(np.sum(counts == len(profiles)))
        no_keys = int(np.sum(counts == 0))
        lines.append(f"    {freq:>4s} MHz: {all_keys} bits always correct, {no_keys} bits always wrong")

    return "\n".join(lines)


def analyze_eigenvalue_fingerprints(profiles):
    """Do keys with similar bit patterns produce similar eigenvalue spectra?"""
    lines = []
    lines.append("\n" + "=" * 70)
    lines.append("3. EIGENVALUE FINGERPRINTING")
    lines.append("=" * 70)

    eigenvalue_matrix = np.array([p["eigenvalues"]["top_20"] for p in profiles])

    # Pairwise cosine similarity
    norms = np.linalg.norm(eigenvalue_matrix, axis=1, keepdims=True)
    normed = eigenvalue_matrix / (norms + 1e-30)
    cos_sim = normed @ normed.T

    lines.append(f"\n  Eigenvalue cosine similarity matrix:")
    lines.append(f"  {'':>6s} " + " ".join(f"K{i:2d}" for i in range(len(profiles))))
    for i in range(len(profiles)):
        row = " ".join(f"{cos_sim[i, j]:.2f}" for j in range(len(profiles)))
        lines.append(f"  K{i:2d}    {row}")

    # Compare with key bit similarity
    bit_sim = np.zeros((len(profiles), len(profiles)))
    for i in range(len(profiles)):
        for j in range(len(profiles)):
            bits_i = profiles[i]["key_bits"]
            bits_j = profiles[j]["key_bits"]
            agreement = sum(a == b for a, b in zip(bits_i, bits_j)) / 256
            bit_sim[i, j] = agreement

    # Correlation between eigenvalue similarity and key similarity
    upper_tri_mask = np.triu_indices(len(profiles), k=1)
    eig_sim_flat = cos_sim[upper_tri_mask]
    bit_sim_flat = bit_sim[upper_tri_mask]
    corr = np.corrcoef(eig_sim_flat, bit_sim_flat)[0, 1]

    lines.append(f"\n  Correlation (eigenvalue similarity vs key bit similarity): {corr:.4f}")
    lines.append(f"  Mean eigenvalue self-similarity: {np.mean(np.diag(cos_sim)):.4f}")
    lines.append(f"  Mean eigenvalue cross-similarity: {np.mean(eig_sim_flat):.4f}")
    lines.append(f"  Mean key bit similarity (random baseline ~0.500): {np.mean(bit_sim_flat):.4f}")

    # Ground state variation
    ground_states = [p["eigenvalues"]["ground_state"] for p in profiles]
    lines.append(f"\n  Ground state energies:")
    lines.append(f"    Mean: {np.mean(ground_states):.4f}")
    lines.append(f"    Std:  {np.std(ground_states):.4f}")
    lines.append(f"    Range: [{min(ground_states):.4f}, {max(ground_states):.4f}]")

    return "\n".join(lines)


def analyze_per_bit_difficulty(profiles):
    """Which bit positions are consistently easy/hard across keys?"""
    lines = []
    lines.append("\n" + "=" * 70)
    lines.append("4. PER-BIT DIFFICULTY (SH READBACK)")
    lines.append("=" * 70)

    all_matches = []
    for p in profiles:
        sh_bits = p["sh_readback"]["bits"]
        target = p["key_bits"]
        matches = [1 if sh_bits[i] == target[i] else 0 for i in range(256)]
        all_matches.append(matches)

    match_matrix = np.array(all_matches)
    success_rate = np.mean(match_matrix, axis=0)

    # Per-block success
    lines.append(f"\n  Per-64-bit block success rates:")
    for start in range(0, 256, 64):
        block_rate = np.mean(success_rate[start:start+64])
        lines.append(f"    Bits {start:3d}-{start+63:3d}: {block_rate:.4f}")

    # Histogram of per-bit success
    hist_bins = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.01]
    counts, _ = np.histogram(success_rate, bins=hist_bins)
    lines.append(f"\n  Per-bit success rate distribution:")
    labels = ["0-10%", "10-30%", "30-50%", "50-70%", "70-90%", "90-100%"]
    for label, count in zip(labels, counts):
        lines.append(f"    {label:>8s}: {count:3d} bits")

    # Chi-squared test: is success rate different from random (50%)?
    from scipy.stats import chisquare
    observed = np.array([np.sum(match_matrix), match_matrix.size - np.sum(match_matrix)])
    expected = np.array([match_matrix.size / 2, match_matrix.size / 2])
    chi2, p_val = chisquare(observed, expected)
    lines.append(f"\n  Chi-squared test (SH readback vs random 50%):")
    lines.append(f"    chi2 = {chi2:.4f}, p = {p_val:.6f}")
    if p_val < 0.001:
        lines.append(f"    SIGNIFICANT: SH readback is significantly better than random")
    else:
        lines.append(f"    Not significant at p<0.001")

    return "\n".join(lines)


def analyze_channel_agreement(profiles):
    """How do different channels agree/disagree?"""
    lines = []
    lines.append("\n" + "=" * 70)
    lines.append("5. CROSS-CHANNEL AGREEMENT")
    lines.append("=" * 70)

    channels = ["sh_readback", "thread_zflip", "thread_vector", "lsq_inversion"]
    channel_labels = ["SH Readback", "Thread Z-Flip", "Thread Vector", "LSQ Inversion"]

    # Pairwise agreement between channels across all keys
    lines.append(f"\n  Average pairwise agreement between channels (across all keys):")

    for ci, c1 in enumerate(channels):
        for cj, c2 in enumerate(channels):
            if cj <= ci:
                continue
            agreements = []
            for p in profiles:
                b1 = p[c1]["bits"]
                b2 = p[c2]["bits"]
                agree = sum(a == b for a, b in zip(b1, b2)) / 256
                agreements.append(agree)
            avg_agree = np.mean(agreements)
            lines.append(f"    {channel_labels[ci]:>15s} vs {channel_labels[cj]:<15s}: {avg_agree:.4f}")

    # Average match rate per channel
    lines.append(f"\n  Average match rate per channel:")
    for c, label in zip(channels, channel_labels):
        rates = [p[c]["match_rate"] for p in profiles]
        lines.append(f"    {label:>15s}: {np.mean(rates):.4f} +/- {np.std(rates):.4f}")

    return "\n".join(lines)


def main():
    profiles = load_profiles()
    print(f"Loaded {len(profiles)} key profiles")

    report_parts = []
    report_parts.append("HARMONIC PROFILE ANALYSIS REPORT")
    report_parts.append(f"Keys analyzed: {len(profiles)}")
    report_parts.append(f"Grid size: {profiles[0].get('energy_stats', {}).get('total_energy', 'N/A')}")
    report_parts.append("")

    part1, per_bit_success, mean_mag = analyze_coefficient_stability(profiles)
    report_parts.append(part1)

    part2 = analyze_frequency_patterns(profiles)
    report_parts.append(part2)

    part3 = analyze_eigenvalue_fingerprints(profiles)
    report_parts.append(part3)

    part4 = analyze_per_bit_difficulty(profiles)
    report_parts.append(part4)

    part5 = analyze_channel_agreement(profiles)
    report_parts.append(part5)

    report = "\n".join(report_parts)

    # Print to console
    print(report)

    # Save to Desktop
    output_path = os.path.expanduser("~/Desktop/profile_analysis_report.txt")
    with open(output_path, "w") as f:
        f.write(report)
    print(f"\nReport saved: {output_path}")


if __name__ == "__main__":
    main()
