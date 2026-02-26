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
    boundscheck=False,
)
def numba_unique(a):
    """Identify unique elements in an array, fast"""

    container = np.zeros(np.max(a) + 1, dtype=np.int32)
    for val in a:
        container[val] += 1

    unique_values = np.nonzero(container)[0]
    unique_counts = container[unique_values]
    return unique_values.astype(np.int32), unique_counts.astype(np.int32)


@njit(
    'float32(uint32[:], int32[:], int32, float32, uint32[:])',
    cache=True,
    fastmath=True,
    error_model='numpy',
    boundscheck=False,
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
    parallel=False,
    fastmath=True,
    error_model='numpy',
    boundscheck=False,
)
def compute_entropies(
    X, Y, all_events, f_values, f_value_counts, cardinality_correction,
):
    """Core entropy computation function"""

    conditional_entropy = 0.0
    background_cond_entropy = 0.0
    full_entropy = 0.0
    class_values, class_counts = numba_unique(Y)

    if not cardinality_correction:
        for k in prange(len(class_counts)):
            class_probability = class_counts[k] / all_events
            full_entropy += -class_probability * np.log(class_probability)

    for f_index in prange(len(f_values)):
        _f_value_counts = f_value_counts[f_index]

        if _f_value_counts == 1:
            continue

        initial_prob = _f_value_counts / all_events
        x_value_subspace = np.where(X == f_values[f_index])

        Y_classes = Y[x_value_subspace].astype(np.uint32)
        subspace_size = x_value_subspace[0].size

        # Right-shift to simulate noise
        Y_classes_spoofed = np.zeros(subspace_size, dtype=np.uint32)
        for enx, el in enumerate(x_value_subspace[0]):
            index = (el + _f_value_counts) % len(Y)
            Y_classes_spoofed[enx] = Y[index]

        nonzero_class_counts = np.zeros(len(class_values), dtype=np.uint32)
        nonzero_class_counts_spoofed = np.zeros(len(class_values), dtype=np.uint32)

        # Cache nonzero counts
        for index, c in enumerate(class_values):
            nonzero_class_counts[index] = np.count_nonzero(Y_classes == c)
            nonzero_class_counts_spoofed[index] = np.count_nonzero(Y_classes_spoofed == c)

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
)
def stratified_subsampling(Y, X, approximation_factor, _f_values_X):

    all_events = len(X)
    final_space_size = int(approximation_factor * all_events)

    unique_samples_per_val = int(final_space_size / len(_f_values_X))

    if unique_samples_per_val == 0:
        return Y, X

    final_index_array = np.empty(final_space_size)

    index_offset = 0
    for fval in _f_values_X:

        # note: this is not randomized due to batch effects, could be an improvement
        x_indices = np.where(X == fval)[0][:unique_samples_per_val]
        x_indices_len = len(x_indices)
        second_offset = (index_offset + x_indices_len)
        final_index_array[index_offset:second_offset] = x_indices
        index_offset += x_indices_len

    final_index_array = final_index_array.astype(np.int32)

    X = X[final_index_array]
    Y = Y[final_index_array]

    return Y, X


@njit(
    'float32(int32[:], int32[:], float32, b1)',
    cache=True,
    fastmath=True,
    error_model='numpy',
    boundscheck=False,
)
def mutual_info_estimator_numba(
    Y, X, approximation_factor=1.0, cardinality_correction=False,
):
    """Core estimator logic. Compute unique elements, subset if required"""

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
