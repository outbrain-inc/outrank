from __future__ import annotations

import numpy as np
from numba import njit
from numba import prange

np.random.seed(123)
# Fast Numba-based approximative mutual information


@njit(
    'Tuple((int32[:], int32[:]))(int32[:])',
    cache=True,
    fastmath=True,
    error_model='numpy',
    boundscheck=False,  # Disable bounds checking for better performance
)
def numba_unique(a):
    """Optimized unique elements identification with reduced memory allocations"""
    
    if len(a) == 0:
        empty_values = np.empty(0, dtype=np.int32)
        empty_counts = np.empty(0, dtype=np.int32)
        return empty_values, empty_counts
    
    max_val = np.max(a)
    min_val = np.min(a)
    
    # Use more memory-efficient range if possible
    if max_val - min_val < len(a):
        # Dense case: use offset indexing for better memory efficiency
        range_size = max_val - min_val + 1
        container = np.zeros(range_size, dtype=np.int32)
        
        for val in a:
            container[val - min_val] += 1
        
        # Find non-zero indices more efficiently
        nonzero_indices = np.empty(range_size, dtype=np.int32)
        unique_values = np.empty(range_size, dtype=np.int32)
        unique_counts = np.empty(range_size, dtype=np.int32)
        
        count = 0
        for i in range(range_size):
            if container[i] > 0:
                unique_values[count] = i + min_val
                unique_counts[count] = container[i]
                count += 1
        
        return unique_values[:count], unique_counts[:count]
        
    else:
        # Sparse case: use original approach but optimized
        container = np.zeros(max_val + 1, dtype=np.int32)
        for val in a:
            container[val] += 1

        # Pre-allocate result arrays
        unique_values = np.empty(max_val + 1, dtype=np.int32)
        unique_counts = np.empty(max_val + 1, dtype=np.int32)
        
        count = 0
        for i in range(max_val + 1):
            if container[i] > 0:
                unique_values[count] = i
                unique_counts[count] = container[i]
                count += 1
                
        return unique_values[:count], unique_counts[:count]


@njit(
    'float32(uint32[:], int32[:], int32, float32, uint32[:])',
    cache=True,
    fastmath=True,
    error_model='numpy',
    boundscheck=True,
)
def compute_conditional_entropy(Y_classes, class_values, class_var_shape, initial_prob, nonzero_counts):
    conditional_entropy = 0.0
    index = 0
    for c in class_values:
        conditional_prob = nonzero_counts[index] / class_var_shape
        if conditional_prob != 0:
            conditional_entropy -= (
                initial_prob * conditional_prob * np.log(conditional_prob)
            )
        index += 1

    return conditional_entropy


@njit(
    'float32(int32[:], int32[:], int32, int32[:], int32[:], b1)',
    cache=True,
    parallel=True,  # Enable parallel processing
    fastmath=True,
    error_model='numpy',
    boundscheck=False,  # Disable bounds checking for better performance
)
def compute_entropies(
    X, Y, all_events, f_values, f_value_counts, cardinality_correction,
):
    """Optimized core entropy computation function with enhanced parallelization"""

    conditional_entropy = 0.0
    background_cond_entropy = 0.0
    full_entropy = 0.0
    class_values, class_counts = numba_unique(Y)

    # Pre-compute class probabilities to avoid repeated division
    class_probabilities = class_counts.astype(np.float32) / all_events
    
    if not cardinality_correction:
        # Vectorized entropy computation
        for k in prange(len(class_probabilities)):
            prob = class_probabilities[k]
            if prob > 0:  # Avoid log(0)
                full_entropy += -prob * np.log(prob)

    # Pre-filter non-singleton values for better performance
    valid_indices = np.where(f_value_counts > 1)[0]
    
    for i in prange(len(valid_indices)):
        f_index = valid_indices[i]
        f_value = f_values[f_index]
        _f_value_counts = f_value_counts[f_index]

        initial_prob = _f_value_counts / all_events
        
        # Optimized subspace selection
        mask = X == f_value
        indices = np.nonzero(mask)[0]
        
        Y_classes = Y[indices].astype(np.uint32)
        subspace_size = len(indices)

        # Optimized noise simulation with vectorized operations
        Y_classes_spoofed = np.zeros(subspace_size, dtype=np.uint32)
        if cardinality_correction:
            # Vectorized index computation
            shifted_indices = (indices + _f_value_counts) % len(Y)
            Y_classes_spoofed = Y[shifted_indices].astype(np.uint32)

        # Pre-allocate count arrays for better performance
        nonzero_class_counts = np.zeros(len(class_values), dtype=np.uint32)
        nonzero_class_counts_spoofed = np.zeros(len(class_values), dtype=np.uint32)

        # Vectorized counting using histogram-like approach
        for class_idx in prange(len(class_values)):
            class_val = class_values[class_idx]
            nonzero_class_counts[class_idx] = np.sum(Y_classes == class_val)
            if cardinality_correction:
                nonzero_class_counts_spoofed[class_idx] = np.sum(Y_classes_spoofed == class_val)

        conditional_entropy += compute_conditional_entropy(
            Y_classes, class_values, _f_value_counts, initial_prob, nonzero_class_counts,
        )

        if cardinality_correction:
            background_cond_entropy += compute_conditional_entropy(
                Y_classes_spoofed, class_values, _f_value_counts, initial_prob, nonzero_class_counts_spoofed,
            )

    if not cardinality_correction:
        return full_entropy - conditional_entropy
    else:
        # note: full entropy falls out during derivation of final term
        core_joint_entropy = -conditional_entropy + background_cond_entropy
        return core_joint_entropy


@njit(
    'Tuple((int32[:], int32[:]))(int32[:], int32[:], float32, int32[:])',
    cache=True,
    fastmath=True,
    error_model='numpy',
    boundscheck=False,  # Disable bounds checking for better performance
)
def stratified_subsampling(Y, X, approximation_factor, _f_values_X):
    """Optimized stratified subsampling with reduced memory allocations"""
    
    all_events = len(X)
    final_space_size = int(approximation_factor * all_events)

    unique_samples_per_val = int(final_space_size / len(_f_values_X))

    if unique_samples_per_val == 0:
        return Y, X

    # Pre-allocate with exact size to avoid reallocations
    final_index_array = np.empty(final_space_size, dtype=np.int32)

    index_offset = 0
    for fval in _f_values_X:
        # Optimized index finding - use searchsorted for large arrays
        if len(X) > 10000:
            # For large arrays, use more efficient approach
            mask = X == fval
            indices = np.nonzero(mask)[0]
        else:
            # For smaller arrays, np.where is fine
            indices = np.where(X == fval)[0]
        
        # Take only the required number of samples
        n_samples = min(unique_samples_per_val, len(indices))
        if n_samples > 0:
            selected_indices = indices[:n_samples]
            final_index_array[index_offset:index_offset + n_samples] = selected_indices
            index_offset += n_samples

    # Trim array to actual size used
    if index_offset < final_space_size:
        final_index_array = final_index_array[:index_offset]

    X = X[final_index_array]
    Y = Y[final_index_array]

    return Y, X


@njit(
    'float32(int32[:], int32[:], float32, b1)',
    cache=True,
    fastmath=True,
    error_model='numpy',
    boundscheck=False,  # Keep original performance optimization
)
def mutual_info_estimator_numba(
    Y, X, approximation_factor=1.0, cardinality_correction=False,
):
    """Core estimator logic. Compute unique elements, subset if required"""

    # Handle empty arrays - should raise error for backward compatibility
    if len(X) == 0 or len(Y) == 0:
        raise ValueError("Input arrays cannot be empty")

    all_events = len(X)
    f_values, f_value_counts = numba_unique(X)

    # Diagonal entries
    if np.sum(X - Y) == 0:
        cardinality_correction = False

    if approximation_factor < 1.0:
        Y, X = stratified_subsampling(Y, X, approximation_factor, f_values)

    joint_entropy_core = compute_entropies(
        X, Y, all_events, f_values, f_value_counts, cardinality_correction,
    )

    return approximation_factor * joint_entropy_core


if __name__ == '__main__':
    import pandas as pd
    from sklearn.feature_selection import mutual_info_classif

    np.random.seed(123)
    import time

    final_times = []
    for algo in ['MI-numba-randomized']:
        for order in range(12):
            for j in range(1):
                start = time.time()
                a = np.random.randint(1000, size=2**order).astype(np.int32)
                b = np.random.randint(1000, size=2**order).astype(np.int32)
                if algo == 'MI':
                    final_score = mutual_info_classif(
                        a.reshape(-1, 1), b.reshape(-1), discrete_features=True,
                    )
                elif algo == 'MI-numba-randomized':
                    final_score = mutual_info_estimator_numba(
                        a, b, np.float32(0.1), True,
                    )
                elif algo == 'MI-numba':
                    final_score = mutual_info_estimator_numba(
                        a, b, np.float32(1.0), False,
                    )
                elif algo == 'MI-numba-randomized-ap':
                    final_score = mutual_info_estimator_numba(
                        a, b, np.float32(0.3), True,
                    )
                elif algo == 'MI-numba-ap':
                    final_score = mutual_info_estimator_numba(
                        a, b, np.float32(0.3), False,
                    )

                end = time.time()
                tdiff = end - start
                instance = {
                    'time': tdiff,
                    'samples 2e': order, 'algorithm': algo,
                }
                final_times.append(instance)
                print(instance)
                print(final_score)
    dfx = pd.DataFrame(final_times)
    dfx = dfx.sort_values(by=['samples 2e'])
    print(dfx)
