#!/usr/bin/env python3
"""
End-to-end example of multivalue feature ranking with OutRank.

This example demonstrates the complete workflow:
1. Loading CSV data with multivalue features (using '_' delimiter)
2. Using OutRank's main API to rank features
3. Comparing multivalue-aware vs traditional approaches

IMPORTANT: Multivalue features use '_' as delimiter (not ',') to avoid 
           conflicts with CSV format!
"""

import sys
import os
import pandas as pd
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from outrank.algorithms.feature_ranking.ranking_mi_multivalue import multivalue_mutual_info_estimator
from outrank.algorithms.importance_estimator import conduct_feature_ranking


def example_basic_usage():
    """Example 1: Basic multivalue MI computation"""
    print("=" * 70)
    print("EXAMPLE 1: Basic Multivalue MI Computation")
    print("=" * 70)
    print()
    
    # Load data with multivalue features (using '_' delimiter)
    data_path = os.path.join(os.path.dirname(__file__), 'multivalue_data.csv')
    df = pd.read_csv(data_path)
    
    print("Sample data (first 5 rows):")
    print(df.head())
    print()
    print(f"Data shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print()
    
    # Extract multivalue features
    interests = df['interests'].values
    skills = df['skills'].values
    purchased = df['purchased'].values
    
    print("Multivalue feature examples:")
    print(f"  interests[0]: {interests[0]}")
    print(f"  skills[0]: {skills[0]}")
    print(f"  purchased[0]: {purchased[0]}")
    print()
    print("NOTE: Values are separated by '_' (not ',') to avoid CSV conflicts!")
    print()
    
    # Compute MI between features using different algorithms
    algorithms = ['jaccard', 'overlap', 'set_based']
    
    print("Computing MI between 'interests' and 'skills':")
    for algo in algorithms:
        # Use delimiter='_' to match our data format
        mi_score = multivalue_mutual_info_estimator(
            interests, skills, algorithm=algo, delimiter='_'
        )
        print(f"  {algo:>10}: {mi_score:.6f}")
    print()
    
    print("Computing MI between 'interests' and 'purchased':")
    for algo in algorithms:
        mi_score = multivalue_mutual_info_estimator(
            interests, purchased, algorithm=algo, delimiter='_'
        )
        print(f"  {algo:>10}: {mi_score:.6f}")
    print()


def example_with_conduct_ranking():
    """Example 2: Using with OutRank's conduct_feature_ranking"""
    print("=" * 70)
    print("EXAMPLE 2: Integration with OutRank's Feature Ranking API")
    print("=" * 70)
    print()
    
    # Load data
    data_path = os.path.join(os.path.dirname(__file__), 'multivalue_data.csv')
    df = pd.read_csv(data_path)
    
    interests = df['interests'].values
    satisfaction = df['satisfaction'].values
    
    print("Ranking 'interests' against 'satisfaction' (target):")
    print()
    
    # Mock args object for different heuristics
    class MockArgs:
        def __init__(self, heuristic):
            self.heuristic = heuristic
            self.mi_stratified_sampling_ratio = 1.0
    
    # Test multivalue heuristics
    heuristics = [
        'MI-multivalue-jaccard',
        'MI-multivalue-overlap', 
        'MI-multivalue-set',
    ]
    
    print("Multivalue MI heuristics:")
    for heuristic in heuristics:
        args = MockArgs(heuristic)
        score = conduct_feature_ranking(interests, satisfaction, args)
        print(f"  {heuristic:>25}: {score:.6f}")
    print()
    
    # Compare with traditional MI (requires encoding)
    print("Traditional MI heuristic (for comparison):")
    # For traditional MI, we need to encode the multivalue feature
    # This is just for demonstration - normally you'd use the multivalue approach
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    interests_encoded = le.fit_transform(interests)
    satisfaction_encoded = le.fit_transform(satisfaction)
    
    args = MockArgs('MI')
    score = conduct_feature_ranking(interests_encoded, satisfaction_encoded, args)
    print(f"  {'MI (encoded)':>25}: {score:.6f}")
    print()


def example_sequential_patterns():
    """Example 3: Handling sequential patterns without direct intersections"""
    print("=" * 70)
    print("EXAMPLE 3: Sequential Patterns (No Direct Intersections)")
    print("=" * 70)
    print()
    
    print("Testing case where row-wise intersections are empty")
    print("but information exists in sequential patterns:")
    print()
    
    # Create data with sequential patterns but no direct intersections
    Col1 = np.array(['a_b', 'b_c', 'c_d', 'd_e', 'e_f'])
    Col2 = np.array(['i_j_k', 'j_k_l', 'k_l_m', 'l_m_n', 'm_n_o'])
    
    print("Col1:", Col1)
    print("Col2:", Col2)
    print()
    print("Note: No value in Col1 appears in Col2 (empty intersections)")
    print("But both have sequential patterns (overlapping values across rows)")
    print()
    
    algorithms = ['jaccard', 'overlap', 'set_based']
    
    print("Results (all algorithms detect information):")
    for algo in algorithms:
        score = multivalue_mutual_info_estimator(Col1, Col2, algorithm=algo, delimiter='_')
        print(f"  {algo:>10}: {score:.6f} ✓")
    print()


def example_delimiter_importance():
    """Example 4: Demonstrating importance of delimiter choice"""
    print("=" * 70)
    print("EXAMPLE 4: Delimiter Choice - Why '_' Instead of ','")
    print("=" * 70)
    print()
    
    print("PROBLEM with ',' delimiter in CSV files:")
    print("-" * 70)
    
    # Show problematic data
    print("\nIf we used ',' as delimiter:")
    print("  CSV line: 1,sports,music,python,sql,high")
    print("  Parsed as: ['1', 'sports', 'music', 'python', 'sql', 'high']")
    print("  → Can't distinguish: is 'sports,music' one field or two?")
    print()
    
    print("SOLUTION with '_' delimiter:")
    print("-" * 70)
    print("\nUsing '_' as delimiter:")
    print("  CSV line: 1,sports_music,python_sql,high")
    print("  Parsed as: ['1', 'sports_music', 'python_sql', 'high']")
    print("  → Clear structure: 'sports_music' is one field with two values")
    print()
    
    # Demonstrate with actual data
    data_with_underscore = np.array(['sports_music', 'tech_art', 'sports_music_tech'])
    
    print("Parsing multivalue features with '_' delimiter:")
    from outrank.algorithms.feature_ranking.ranking_mi_multivalue import parse_multivalue_feature
    
    parsed = parse_multivalue_feature(data_with_underscore, delimiter='_')
    for i, (original, parsed_set) in enumerate(zip(data_with_underscore, parsed)):
        print(f"  '{original}' → {parsed_set}")
    print()
    
    print("✓ Clean parsing! Each multivalue feature is correctly split.")
    print()


def example_comparison_with_expansion():
    """Example 5: Multivalue MI vs One-Hot Expansion"""
    print("=" * 70)
    print("EXAMPLE 5: Multivalue MI vs Traditional One-Hot Expansion")
    print("=" * 70)
    print()
    
    # Load data
    data_path = os.path.join(os.path.dirname(__file__), 'multivalue_data.csv')
    df = pd.read_csv(data_path)
    
    interests = df['interests'].values[:10]  # Use subset for clarity
    skills = df['skills'].values[:10]
    
    print("Original multivalue data (first 5 rows):")
    for i in range(5):
        print(f"  Row {i}: interests='{interests[i]}', skills='{skills[i]}'")
    print()
    
    # Traditional approach: expand to one-hot
    print("TRADITIONAL APPROACH: One-hot expansion")
    print("-" * 70)
    
    # Get all unique values
    all_interest_values = set()
    all_skill_values = set()
    for interest, skill in zip(interests, skills):
        all_interest_values.update(interest.split('_'))
        all_skill_values.update(skill.split('_'))
    
    print(f"  Unique interest values: {sorted(all_interest_values)}")
    print(f"  Unique skill values: {sorted(all_skill_values)}")
    print(f"  → Creates {len(all_interest_values)} + {len(all_skill_values)} = "
          f"{len(all_interest_values) + len(all_skill_values)} binary features!")
    print("  → High-dimensional, sparse representation")
    print("  → Loses co-occurrence information within each instance")
    print()
    
    # New approach: direct multivalue MI
    print("NEW APPROACH: Direct multivalue MI")
    print("-" * 70)
    
    mi_score = multivalue_mutual_info_estimator(
        interests, skills, algorithm='set_based', delimiter='_'
    )
    print(f"  Direct MI computation: {mi_score:.6f}")
    print("  → Works directly on multivalue features")
    print("  → Preserves co-occurrence information")
    print("  → No expansion needed!")
    print()


def main():
    """Run all examples"""
    print("\n")
    print("#" * 70)
    print("# OutRank Multivalue Features - End-to-End Examples")
    print("#" * 70)
    print("\n")
    
    print("This demonstration shows how to use multivalue features with OutRank.")
    print("Key point: Use '_' as delimiter (not ',') to avoid CSV conflicts!")
    print("\n")
    
    try:
        example_basic_usage()
        example_with_conduct_ranking()
        example_sequential_patterns()
        example_delimiter_importance()
        example_comparison_with_expansion()
        
        print("=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print()
        print("✓ All examples completed successfully!")
        print()
        print("Key takeaways:")
        print("  1. Use '_' as delimiter for multivalue features (not ',')")
        print("  2. Three algorithms available: jaccard, overlap, set_based")
        print("  3. Works with OutRank's main API (conduct_feature_ranking)")
        print("  4. Handles sequential patterns without direct intersections")
        print("  5. No expansion needed - direct computation on multivalue features")
        print()
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
