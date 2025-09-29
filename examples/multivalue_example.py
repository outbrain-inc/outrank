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
    
    # Sample data where each cell contains multiple values separated by commas
    data = {
        'user_id': [1, 2, 3, 4, 5, 6, 7, 8],
        'interests': ['sports,music', 'music,tech', 'sports,tech', 'music,art', 
                     'sports,art', 'tech,art', 'sports,music,tech', 'music,art,tech'],
        'skills': ['python,sql', 'java,sql', 'python,java', 'r,python',
                  'sql,r', 'java,r', 'python,sql,java', 'r,python,sql'],
        'purchased': ['laptop,phone', 'phone,tablet', 'laptop,tablet', 'phone,headphones',
                     'laptop,headphones', 'tablet,headphones', 'laptop,phone,tablet', 'phone,headphones,tablet']
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
        mi_score = multivalue_mutual_info_estimator(interests, skills, algorithm=algo)
        print(f"  {algo:>10}: {mi_score:.6f}")
    
    print("\nMultivalue MI between 'interests' and 'purchased':")
    for algo in algorithms:
        mi_score = multivalue_mutual_info_estimator(interests, purchased, algorithm=algo)
        print(f"  {algo:>10}: {mi_score:.6f}")
    
    print("\nMultivalue MI between 'skills' and 'purchased':")
    for algo in algorithms:
        mi_score = multivalue_mutual_info_estimator(skills, purchased, algorithm=algo)
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
    
    # Create data with clear functional relationship
    functional_x = np.array(['a,b', 'b,c', 'c,d', 'a,b', 'b,c', 'c,d'])
    functional_y = np.array(['x,y', 'y,z', 'z,w', 'x,y', 'y,z', 'z,w'])
    
    print("Functional relationship (X -> Y mapping):")
    for algo in algorithms:
        mi_score = multivalue_mutual_info_estimator(functional_x, functional_y, algorithm=algo)
        print(f"  {algo:>10}: {mi_score:.6f}")
    
    # Compare with independent features
    independent_x = np.array(['a,b', 'c,d', 'e,f', 'g,h', 'i,j', 'k,l'])
    independent_y = np.array(['x,y', 'z,w', 'p,q', 'r,s', 't,u', 'v,w'])
    
    print("\nIndependent features:")
    for algo in algorithms:
        mi_score = multivalue_mutual_info_estimator(independent_x, independent_y, algorithm=algo)
        print(f"  {algo:>10}: {mi_score:.6f}")
    
    print("\n5. Algorithm descriptions:")
    print("  - jaccard:   Uses Jaccard similarity between multivalue sets")
    print("  - overlap:   Based on set overlap and size relationships")
    print("  - set_based: Direct set-based mutual information computation")
    print("\nThe set_based algorithm typically provides the most meaningful")
    print("mutual information scores for multivalue features.")


if __name__ == '__main__':
    main()