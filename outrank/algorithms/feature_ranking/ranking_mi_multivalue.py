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
    
    This approach analyzes the structural patterns in how values co-occur across rows,
    rather than just looking at row-wise intersections. It captures information even
    when X and Y values never directly intersect within the same row.
    """
    if len(X_multivalue) != len(Y_multivalue) or len(X_multivalue) == 0:
        return 0.0
    
    n_samples = len(X_multivalue)
    
    # Build value-to-row mappings to analyze structural patterns
    x_value_to_rows = {}
    y_value_to_rows = {}
    
    for idx, (x_set, y_set) in enumerate(zip(X_multivalue, Y_multivalue)):
        for x_val in x_set:
            if x_val not in x_value_to_rows:
                x_value_to_rows[x_val] = set()
            x_value_to_rows[x_val].add(idx)
        for y_val in y_set:
            if y_val not in y_value_to_rows:
                y_value_to_rows[y_val] = set()
            y_value_to_rows[y_val].add(idx)
    
    # Compute row-level features based on value distribution patterns
    x_features = []
    y_features = []
    
    for i in range(n_samples):
        x_set = X_multivalue[i]
        y_set = Y_multivalue[i]
        
        # Feature 1: Compute Jaccard similarity of row sets themselves
        row_jaccard = 0.0
        if len(x_set) > 0 or len(y_set) > 0:
            intersection = len(x_set.intersection(y_set))
            union = len(x_set.union(y_set))
            row_jaccard = intersection / union if union > 0 else 0.0
        
        # Feature 2: Analyze co-occurrence structure - how much do the values
        # in this row overlap with values from the other feature in adjacent rows
        x_neighbor_overlap = 0.0
        y_neighbor_overlap = 0.0
        
        # Look at neighboring rows (within a window)
        window = 2
        for offset in range(-window, window + 1):
            neighbor_idx = i + offset
            if neighbor_idx >= 0 and neighbor_idx < n_samples and neighbor_idx != i:
                # Check how X values from current row relate to Y values in neighbor
                if neighbor_idx < len(Y_multivalue):
                    neighbor_y = Y_multivalue[neighbor_idx]
                    for x_val in x_set:
                        for y_val in neighbor_y:
                            # Measure co-occurrence strength
                            x_rows = x_value_to_rows.get(x_val, set())
                            y_rows = y_value_to_rows.get(y_val, set())
                            if x_rows and y_rows:
                                jaccard = len(x_rows.intersection(y_rows)) / len(x_rows.union(y_rows))
                                x_neighbor_overlap += jaccard
                
                # Check how Y values from current row relate to X values in neighbor
                if neighbor_idx < len(X_multivalue):
                    neighbor_x = X_multivalue[neighbor_idx]
                    for y_val in y_set:
                        for x_val in neighbor_x:
                            x_rows = x_value_to_rows.get(x_val, set())
                            y_rows = y_value_to_rows.get(y_val, set())
                            if x_rows and y_rows:
                                jaccard = len(x_rows.intersection(y_rows)) / len(x_rows.union(y_rows))
                                y_neighbor_overlap += jaccard
        
        # Normalize by set sizes and window
        normalizer = max(len(x_set) * len(y_set) * window * 2, 1)
        x_neighbor_overlap /= normalizer
        y_neighbor_overlap /= normalizer
        
        # Create composite features
        # Use combination of direct similarity and structural patterns
        x_feature = int(row_jaccard * 10) * 10 + int(x_neighbor_overlap * 100)
        y_feature = int(row_jaccard * 10) * 10 + int(y_neighbor_overlap * 100)
        
        x_features.append(x_feature)
        y_features.append(y_feature)
    
    # Compute MI between the derived features
    mi_score = _compute_discrete_mi(x_features, y_features)
    
    # If MI is still 0, use the set-based approach as it's more robust
    if mi_score == 0.0:
        return set_based_mutual_info(X_multivalue, Y_multivalue)
    
    return mi_score


def multivalue_mi_with_overlap(X_multivalue: List[Set], Y_multivalue: List[Set]) -> float:
    """
    Compute mutual information between multivalue features considering set overlaps.
    
    This algorithm considers both row-wise overlaps and cross-row value 
    co-occurrence patterns to capture information even when row-wise 
    intersections are empty.
    """
    if len(X_multivalue) != len(Y_multivalue) or len(X_multivalue) == 0:
        return 0.0
    
    n_samples = len(X_multivalue)
    
    # Collect all unique values and their row occurrences
    all_x_values = set()
    all_y_values = set()
    x_value_to_rows = {}
    y_value_to_rows = {}
    
    for idx, (x_set, y_set) in enumerate(zip(X_multivalue, Y_multivalue)):
        for x_val in x_set:
            all_x_values.add(x_val)
            if x_val not in x_value_to_rows:
                x_value_to_rows[x_val] = set()
            x_value_to_rows[x_val].add(idx)
        for y_val in y_set:
            all_y_values.add(y_val)
            if y_val not in y_value_to_rows:
                y_value_to_rows[y_val] = set()
            y_value_to_rows[y_val].add(idx)
    
    # Create overlap-based features considering global patterns
    overlap_features = []
    pattern_features = []
    
    for i in range(n_samples):
        x_set = X_multivalue[i]
        y_set = Y_multivalue[i]
        
        # Row-wise overlap features
        intersection_size = len(x_set.intersection(y_set))
        union_size = len(x_set.union(y_set))
        x_size = len(x_set)
        y_size = len(y_set)
        
        # Cross-row co-occurrence: how often do values from this row's X 
        # appear with values from this row's Y in ANY row?
        cross_row_overlap = 0
        for x_val in x_set:
            for y_val in all_y_values:
                if y_val in y_value_to_rows:
                    x_rows = x_value_to_rows.get(x_val, set())
                    y_rows = y_value_to_rows[y_val]
                    # Count shared rows
                    shared = len(x_rows.intersection(y_rows))
                    if shared > 0:
                        cross_row_overlap += shared
        
        # Normalize cross-row overlap
        max_possible_overlap = max(x_size * len(all_y_values), 1)
        normalized_overlap = cross_row_overlap / max_possible_overlap
        
        # Compute composite feature combining row-wise and cross-row information
        if union_size == 0:
            overlap_feature = 0
        else:
            # Direct overlap ratio
            direct_ratio = intersection_size / union_size
            # Combine with cross-row information
            overlap_feature = int(direct_ratio * 5) * 10 + int(normalized_overlap * 10)
        
        # Pattern feature based on value co-occurrence structure
        pattern_feature = int(normalized_overlap * 20) + (y_size % 5)
        
        overlap_features.append(overlap_feature)
        pattern_features.append(pattern_feature)
    
    # Compute MI between overlap features and pattern features
    return _compute_discrete_mi(overlap_features, pattern_features)


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