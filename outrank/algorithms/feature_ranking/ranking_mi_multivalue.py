from __future__ import annotations

import numpy as np
from numba import njit
from numba import prange
from typing import List, Set, Any
import pandas as pd

np.random.seed(123)
# Multivalue Mutual Information algorithms


def _compute_discrete_mi(X: List, Y: List) -> float:
    """
    Compute mutual information between two discrete variables.
    
    This is a simplified implementation that doesn't require external dependencies.
    """
    if len(X) != len(Y) or len(X) == 0:
        return 0.0
    
    n_samples = len(X)
    
    # Count joint and marginal frequencies
    joint_counts = {}
    x_counts = {}
    y_counts = {}
    
    for x, y in zip(X, Y):
        joint_key = (x, y)
        joint_counts[joint_key] = joint_counts.get(joint_key, 0) + 1
        x_counts[x] = x_counts.get(x, 0) + 1
        y_counts[y] = y_counts.get(y, 0) + 1
    
    # Compute mutual information
    mi = 0.0
    for joint_key, joint_count in joint_counts.items():
        x, y = joint_key
        
        p_xy = joint_count / n_samples
        p_x = x_counts[x] / n_samples
        p_y = y_counts[y] / n_samples
        
        if p_xy > 0 and p_x > 0 and p_y > 0:
            mi += p_xy * np.log(p_xy / (p_x * p_y))
    
    return float(mi)


def set_based_mutual_info(X_sets: List[Set], Y_sets: List[Set]) -> float:
    """
    Compute mutual information between two multivalue features represented as sets.
    
    X_sets: List of sets representing multivalue features
    Y_sets: List of sets representing multivalue features
    
    This implements Set-based Mutual Information which treats multivalue features
    as sets and computes MI based on set intersections and unions.
    """
    if len(X_sets) != len(Y_sets) or len(X_sets) == 0:
        return 0.0
    
    n_samples = len(X_sets)
    
    # Compute joint probabilities based on set relationships
    joint_counts = {}
    x_counts = {}
    y_counts = {}
    
    for i in range(n_samples):
        x_set = X_sets[i]
        y_set = Y_sets[i]
        
        # Create hash keys for sets (convert to sorted tuples for hashing)
        x_key = tuple(sorted(x_set)) if x_set else ()
        y_key = tuple(sorted(y_set)) if y_set else ()
        joint_key = (x_key, y_key)
        
        # Count occurrences
        joint_counts[joint_key] = joint_counts.get(joint_key, 0) + 1
        x_counts[x_key] = x_counts.get(x_key, 0) + 1  
        y_counts[y_key] = y_counts.get(y_key, 0) + 1
    
    # Compute mutual information
    mi = 0.0
    for joint_key, joint_count in joint_counts.items():
        x_key, y_key = joint_key
        
        p_xy = joint_count / n_samples
        p_x = x_counts[x_key] / n_samples
        p_y = y_counts[y_key] / n_samples
        
        if p_xy > 0 and p_x > 0 and p_y > 0:
            mi += p_xy * np.log(p_xy / (p_x * p_y))
    
    return float(mi)


def jaccard_based_mutual_info(X_multivalue: List[Set], Y_multivalue: List[Set]) -> float:
    """
    Compute mutual information between two multivalue features using Jaccard similarity.
    
    This approach uses Jaccard similarity coefficients to measure relationships
    between multivalue features, then computes MI based on these similarities.
    """
    if len(X_multivalue) != len(Y_multivalue) or len(X_multivalue) == 0:
        return 0.0
    
    n_samples = len(X_multivalue)
    
    # Compute Jaccard similarities
    jaccard_similarities = []
    for i in range(n_samples):
        x_set = X_multivalue[i]
        y_set = Y_multivalue[i]
        
        if len(x_set) == 0 and len(y_set) == 0:
            jaccard_sim = 1.0  # Both empty sets are considered identical
        elif len(x_set) == 0 or len(y_set) == 0:
            jaccard_sim = 0.0  # One empty, one non-empty
        else:
            intersection = len(x_set.intersection(y_set))
            union = len(x_set.union(y_set))
            jaccard_sim = intersection / union if union > 0 else 0.0
        
        jaccard_similarities.append(jaccard_sim)
    
    # Discretize Jaccard similarities into bins for MI computation
    # Use 10 bins from 0 to 1
    n_bins = 10
    jaccard_binned = np.digitize(jaccard_similarities, bins=np.linspace(0, 1, n_bins))
    
    # Create artificial Y values based on patterns in original Y sets
    y_patterns = []
    for y_set in Y_multivalue:
        # Create a simple pattern representation
        if len(y_set) == 0:
            pattern = 0
        else:
            pattern = len(y_set) % 5  # Simple pattern based on set size
        y_patterns.append(pattern)
    
    # Compute MI between binned Jaccard similarities and Y patterns
    return _compute_discrete_mi(jaccard_binned, y_patterns)


def multivalue_mi_with_overlap(X_multivalue: List[Set], Y_multivalue: List[Set]) -> float:
    """
    Compute mutual information between multivalue features considering set overlaps.
    
    This algorithm creates overlap-based features and computes traditional MI
    on these derived features.
    """
    if len(X_multivalue) != len(Y_multivalue) or len(X_multivalue) == 0:
        return 0.0
    
    n_samples = len(X_multivalue)
    
    # Create overlap-based features
    overlap_features = []
    y_size_features = []
    
    for i in range(n_samples):
        x_set = X_multivalue[i]
        y_set = Y_multivalue[i]
        
        # Overlap-based features
        intersection_size = len(x_set.intersection(y_set))
        union_size = len(x_set.union(y_set))
        x_size = len(x_set)
        y_size = len(y_set)
        
        # Create a composite feature based on set relationships
        if union_size == 0:
            overlap_feature = 0
        else:
            # Encode relationship as: intersection_ratio * 10 + size_difference_ratio
            intersection_ratio = intersection_size / union_size
            size_diff = abs(x_size - y_size)
            max_size = max(x_size, y_size, 1)
            size_diff_ratio = size_diff / max_size
            
            overlap_feature = int(intersection_ratio * 10) * 10 + int(size_diff_ratio * 10)
        
        overlap_features.append(overlap_feature)
        y_size_features.append(y_size)
    
    # Compute MI between overlap features and Y size features
    return _compute_discrete_mi(overlap_features, y_size_features)


def parse_multivalue_feature(feature_vector: np.ndarray, delimiter: str = ',') -> List[Set]:
    """
    Parse a multivalue feature vector into a list of sets.
    
    Args:
        feature_vector: Array of strings where each element contains multiple values
        delimiter: Character used to separate values within each element
    
    Returns:
        List of sets, one for each row in the feature vector
    """
    multivalue_sets = []
    
    for value in feature_vector:
        if pd.isna(value) or value == '' or value == 'nan':
            multivalue_sets.append(set())
        else:
            # Split by delimiter and create set, filtering out empty strings
            value_set = set(str(value).split(delimiter))
            value_set = {v.strip() for v in value_set if v.strip()}
            multivalue_sets.append(value_set)
    
    return multivalue_sets


def multivalue_mutual_info_estimator(
    X_feature: np.ndarray, Y_feature: np.ndarray, 
    algorithm: str = 'jaccard', delimiter: str = ','
) -> float:
    """
    Main entry point for multivalue mutual information computation.
    
    Args:
        X_feature: First multivalue feature (array of strings with delimited values)
        Y_feature: Second multivalue feature (array of strings with delimited values)  
        algorithm: Algorithm to use ('jaccard', 'overlap', 'set_based')
        delimiter: Delimiter used to separate values within each feature
    
    Returns:
        Mutual information score between the two multivalue features
    """
    if len(X_feature) != len(Y_feature) or len(X_feature) == 0:
        return 0.0
    
    # Parse multivalue features into sets
    X_sets = parse_multivalue_feature(X_feature, delimiter)
    Y_sets = parse_multivalue_feature(Y_feature, delimiter)
    
    # Apply selected algorithm
    if algorithm == 'jaccard':
        return jaccard_based_mutual_info(X_sets, Y_sets)
    elif algorithm == 'overlap':
        return multivalue_mi_with_overlap(X_sets, Y_sets)
    elif algorithm == 'set_based':
        return set_based_mutual_info(X_sets, Y_sets)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")