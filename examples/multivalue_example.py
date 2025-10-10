#!/usr/bin/env python3
"""
Example demonstrating multivalue feature ranking with OutRank.

This example shows how to use the new multivalue mutual information algorithms
that make multivalue features first-class citizens in OutRank, without needing
to expand them into one-hot encodings.
"""

import numpy as np
import pandas as pd
from outrank.algorithms.feature_ranking.ranking_mi_multivalue import multivalue_mutual_info_estimator
from outrank.algorithms.importance_estimator import conduct_feature_ranking


def main():
    print("OutRank Multivalue Features Example")
    print("=" * 40)
    
    # Create sample multivalue data
    print("\n1. Creating sample multivalue data...")
    
    # Sample data where each cell contains multiple values separated by underscores
    # NOTE: We use '_' instead of ',' to avoid conflicts with CSV format!
    data = {
        'user_id': [1, 2, 3, 4, 5, 6, 7, 8],
        'interests': ['sports_music', 'music_tech', 'sports_tech', 'music_art', 
                     'sports_art', 'tech_art', 'sports_music_tech', 'music_art_tech'],
        'skills': ['python_sql', 'java_sql', 'python_java', 'r_python',
                  'sql_r', 'java_r', 'python_sql_java', 'r_python_sql'],
        'purchased': ['laptop_phone', 'phone_tablet', 'laptop_tablet', 'phone_headphones',
                     'laptop_headphones', 'tablet_headphones', 'laptop_phone_tablet', 'phone_headphones_tablet']
    }
    
    df = pd.DataFrame(data)
    print("Sample data:")
    print(df)
    
    # Demonstrate traditional approach vs new multivalue approach
    print("\n2. Comparing approaches...")
    
    interests = df['interests'].values
    skills = df['skills'].values
    purchased = df['purchased'].values
    
    # Test different multivalue MI algorithms
    algorithms = ['jaccard', 'overlap', 'set_based']
    
    print("\nMultivalue MI between 'interests' and 'skills':")
    for algo in algorithms:
        mi_score = multivalue_mutual_info_estimator(interests, skills, algorithm=algo, delimiter='_')
        print(f"  {algo:>10}: {mi_score:.6f}")
    
    print("\nMultivalue MI between 'interests' and 'purchased':")
    for algo in algorithms:
        mi_score = multivalue_mutual_info_estimator(interests, purchased, algorithm=algo, delimiter='_')
        print(f"  {algo:>10}: {mi_score:.6f}")
    
    print("\nMultivalue MI between 'skills' and 'purchased':")
    for algo in algorithms:
        mi_score = multivalue_mutual_info_estimator(skills, purchased, algorithm=algo, delimiter='_')
        print(f"  {algo:>10}: {mi_score:.6f}")
    
    # Demonstrate usage with OutRank's importance estimator
    print("\n3. Using with OutRank's importance estimator...")
    
    class MockArgs:
        def __init__(self, heuristic):
            self.heuristic = heuristic
            self.mi_stratified_sampling_ratio = 1.0
    
    heuristics = ['MI-multivalue-jaccard', 'MI-multivalue-overlap', 'MI-multivalue-set']
    
    print("\nFeature ranking between 'interests' and 'purchased':")
    for heuristic in heuristics:
        args = MockArgs(heuristic)
        score = conduct_feature_ranking(interests, purchased, args)
        print(f"  {heuristic:>25}: {score:.6f}")
    
    # Demonstrate functional relationship detection
    print("\n4. Functional relationship detection...")
    
    # Create data with clear functional relationship (using '_' delimiter)
    functional_x = np.array(['a_b', 'b_c', 'c_d', 'a_b', 'b_c', 'c_d'])
    functional_y = np.array(['x_y', 'y_z', 'z_w', 'x_y', 'y_z', 'z_w'])
    
    print("Functional relationship (X -> Y mapping):")
    for algo in algorithms:
        mi_score = multivalue_mutual_info_estimator(functional_x, functional_y, algorithm=algo, delimiter='_')
        print(f"  {algo:>10}: {mi_score:.6f}")
    
    # Compare with independent features
    independent_x = np.array(['a_b', 'c_d', 'e_f', 'g_h', 'i_j', 'k_l'])
    independent_y = np.array(['x_y', 'z_w', 'p_q', 'r_s', 't_u', 'v_w'])
    
    print("\nIndependent features:")
    for algo in algorithms:
        mi_score = multivalue_mutual_info_estimator(independent_x, independent_y, algorithm=algo, delimiter='_')
        print(f"  {algo:>10}: {mi_score:.6f}")
    
    print("\n5. Algorithm descriptions:")
    print("  - jaccard:   Analyzes structural patterns via neighbor row analysis")
    print("               and value co-occurrence, works even with empty intersections")
    print("  - overlap:   Based on set overlap and cross-row co-occurrence patterns")
    print("  - set_based: Direct set-based mutual information computation")
    print("\nAll algorithms now properly handle cases where row-wise intersections")
    print("are empty but structural patterns exist (e.g., sequential patterns).")
    print("The set_based algorithm typically provides the most meaningful")
    print("mutual information scores for multivalue features.")
    print("\nIMPORTANT: Default delimiter is '_' (not ',') to avoid CSV conflicts!")
    
    # Demonstrate the fix for the GitHub issue
    print("\n6. Handling sequential patterns (GitHub issue fix):")
    sequential_x = np.array(['a_b', 'b_c', 'c_d', 'd_e', 'e_f'])
    sequential_y = np.array(['i_j_k', 'j_k_l', 'k_l_m', 'l_m_n', 'm_n_o'])
    
    print("Data with NO row-wise intersections but clear sequential patterns:")
    print("  X:", sequential_x[:3], "...")
    print("  Y:", sequential_y[:3], "...")
    print("\nResults (all algorithms now detect information):")
    for algo in algorithms:
        mi_score = multivalue_mutual_info_estimator(sequential_x, sequential_y, algorithm=algo, delimiter='_')
        print(f"  {algo:>10}: {mi_score:.6f} ✓")


if __name__ == '__main__':
    main()