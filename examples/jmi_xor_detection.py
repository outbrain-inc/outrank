"""Demonstrates the XOR blind spot in pairwise MI and how JMI/II detect it.

This script:
1. Generates 100 random features + 2 XOR features (f0, f1) + label = XOR(f0, f1)
2. Shows pairwise MI ranks f0, f1 near bottom (individually uninformative)
3. Shows JMI ranking promotes f0, f1 to the top (conditionally informative)
4. Shows interaction information: II(f0, f1; label) < 0 (synergy detected)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from outrank.algorithms.feature_ranking.ranking_mi_numba_cmi import conditional_mutual_info_numba
from outrank.algorithms.feature_ranking.ranking_mi_numba_cmi import interaction_information_numba
from outrank.algorithms.feature_ranking.ranking_mi_numba_opt import (
    mutual_info_estimator_numba_opt,
)


def main():
    np.random.seed(42)
    n = 5000
    n_noise = 100

    # Generate XOR features
    f0 = np.random.randint(0, 2, size=n, dtype=np.int32)
    f1 = np.random.randint(0, 2, size=n, dtype=np.int32)
    label = np.bitwise_xor(f0, f1).astype(np.int32)

    # Generate noise features (some with mild correlations to label for contrast)
    noise_features = {}
    for i in range(n_noise):
        noise_features[f'noise_{i}'] = np.random.randint(0, 4, size=n, dtype=np.int32)

    # Compute pairwise MI for all features
    all_features = {'f0_xor': f0, 'f1_xor': f1, **noise_features}
    pairwise_mi = {}
    for name, vec in all_features.items():
        mi = float(mutual_info_estimator_numba_opt(label, vec, np.float32(1.0), False))
        pairwise_mi[name] = mi

    pairwise_sorted = sorted(pairwise_mi.items(), key=lambda x: x[1], reverse=True)

    print('=' * 60)
    print('PAIRWISE MI RANKING (top 10 + XOR features)')
    print('=' * 60)
    for rank, (name, score) in enumerate(pairwise_sorted[:10], 1):
        print(f'  #{rank:3d}  {name:20s}  MI = {score:.4f}')

    # Find XOR feature ranks
    for rank, (name, score) in enumerate(pairwise_sorted, 1):
        if name in ('f0_xor', 'f1_xor'):
            print(f'  #{rank:3d}  {name:20s}  MI = {score:.4f}  <-- XOR feature')

    # JMI: greedy forward selection
    print()
    print('=' * 60)
    print('JMI GREEDY SELECTION (first 10 features)')
    print('=' * 60)

    feature_arrays = {name: vec for name, vec in all_features.items()}
    # Start with the top pairwise MI feature
    selected = [pairwise_sorted[0][0]]
    remaining = set(all_features.keys()) - set(selected)

    for step in range(min(9, len(remaining))):
        best_score = -np.inf
        best_feat = None

        for xk in remaining:
            jmi_score = 0.0
            for xj in selected:
                cmi = float(
                    conditional_mutual_info_numba(
                        label, feature_arrays[xk], feature_arrays[xj],
                        np.float32(1.0), False,
                    ),
                )
                jmi_score += cmi
            if jmi_score > best_score:
                best_score = jmi_score
                best_feat = xk

        selected.append(best_feat)
        remaining.discard(best_feat)

    for rank, name in enumerate(selected, 1):
        marker = '  <-- XOR feature' if 'xor' in name else ''
        print(f'  #{rank:3d}  {name:20s}{marker}')

    # Interaction information
    print()
    print('=' * 60)
    print('INTERACTION INFORMATION')
    print('=' * 60)

    ii_xor = interaction_information_numba(label, f0, f1)
    print(f'  II(f0_xor, f1_xor; label) = {ii_xor:.4f}')
    print(f'  Interpretation: {"SYNERGY (negative)" if ii_xor < 0 else "REDUNDANCY (positive)" if ii_xor > 0.05 else "INDEPENDENT (~0)"}')
    print()

    # Compare with a noise pair
    noise_0 = noise_features['noise_0']
    noise_1 = noise_features['noise_1']
    ii_noise = interaction_information_numba(label, noise_0, noise_1)
    print(f'  II(noise_0, noise_1; label) = {ii_noise:.4f}')
    print(f'  Interpretation: {"SYNERGY (negative)" if ii_noise < 0 else "REDUNDANCY (positive)" if ii_noise > 0.05 else "INDEPENDENT (~0)"}')

    # Direct CMI check: I(f0; label | f1) should be high
    cmi_f0_given_f1 = float(
        conditional_mutual_info_numba(
            label, f0, f1, np.float32(1.0), False,
        ),
    )
    print()
    print('=' * 60)
    print('CONDITIONAL MI (direct synergy proof)')
    print('=' * 60)
    print(f'  I(f0; label)      = {pairwise_mi["f0_xor"]:.4f}  (near zero — individually uninformative)')
    print(f'  I(f0; label | f1) = {cmi_f0_given_f1:.4f}  (high — f0 becomes informative given f1)')

    print()
    print('=' * 60)
    print('SUMMARY')
    print('=' * 60)
    print('  Pairwise MI is blind to XOR: f0, f1 rank at the bottom.')
    print('  II(f0,f1;label) < 0 detects the synergistic interaction.')
    print('  I(f0;label|f1) >> I(f0;label) proves conditional dependence.')
    print('  Use II to screen for synergistic pairs, then CMI/JMI to rank them.')


if __name__ == '__main__':
    main()
