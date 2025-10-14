from __future__ import annotations
import numpy as np
from numba import njit, prange

np.random.seed(123)


@njit('Tuple((int32[:], int32[:]))(int32[:])', cache=True, fastmath=True)
def numba_unique(a):
    """
    Identify unique elements and their counts in a non-negative integer array.
    This version finds the max value in one pass to size the container.
    """
    maxv = 0
    if a.size > 0:
        for i in range(a.size):
            if a[i] > maxv:
                maxv = a[i]
    container = np.zeros(maxv + 1, dtype=np.int32)
    for i in range(a.size):
        container[a[i]] += 1
    unique_values = np.nonzero(container)[0].astype(np.int32)
    unique_counts = container[unique_values].astype(np.int32)
    return unique_values, unique_counts


@njit('float32(float32, int32, uint32[:])', cache=True, fastmath=True)
def compute_conditional_entropy(initial_prob, group_size, class_counts):
    """
    Calculates the contribution to conditional entropy for a single group.
    - initial_prob: P(X=v)
    - group_size: Number of elements in this group.
    - class_counts: Histogram of Y classes within this group.
    """
    ce = 0.0
    inv_group_size = 1.0 / group_size
    for count in class_counts:
        if count > 0:
            conditional_prob = count * inv_group_size
            ce -= initial_prob * conditional_prob * np.log(conditional_prob)
    return ce


@njit('Tuple((int32[:], int32[:], int32[:], int32[:]))(int32[:])', cache=True, fastmath=True)
def build_groups(X):
    """
    Pre-processes X to create an efficient grouping structure.
    This avoids repeated np.where scans.
    Returns:
    - f_values: Unique values in X.
    - f_counts: Counts of each unique value.
    - group_starts: Start indices for each group in the `positions` array.
    - positions: A single array of indices [0..N-1], sorted by the value of X at that index.
    """
    f_values, f_counts = numba_unique(X)
    V = f_values.size

    vmax = 0
    if V > 0:
        for i in range(V):
            if f_values[i] > vmax:
                vmax = f_values[i]
    value_to_group_idx = np.full(vmax + 1, -1, dtype=np.int32)
    for i in range(V):
        value_to_group_idx[f_values[i]] = i

    group_starts = np.zeros(V, dtype=np.int32)
    run = 0
    for i in range(V):
        group_starts[i] = run
        run += f_counts[i]

    positions = np.empty(X.size, dtype=np.int32)
    cursors = group_starts.copy()
    for i in range(X.size):
        xi = X[i]
        gi = value_to_group_idx[xi]
        pos = cursors[gi]
        positions[pos] = i
        cursors[gi] = pos + 1

    return f_values, f_counts, group_starts, positions


@njit(
    'float32(int32[:], int32, int32[:], int32[:], int32[:], int32[:], b1)',
    cache=True,
    fastmath=True,
)
def compute_entropies_grouped(
    Y, all_events,
    f_values, f_counts, group_starts, positions,
    cardinality_correction,
):
    """
    Core entropy computation using the pre-built grouping structure.
    This is much faster as it avoids scans and temporary arrays in the loop.
    """
    class_values, class_counts = numba_unique(Y)
    C = class_values.size

    full_entropy = 0.0
    if not cardinality_correction:
        invN = 1.0 / all_events
        for k in range(class_counts.size):
            p = class_counts[k] * invN
            if p > 0.0:
                full_entropy -= p * np.log(p)

    cmax = 0
    if C > 0:
        for i in range(C):
            if class_values[i] > cmax:
                cmax = class_values[i]
    class_to_idx = np.full(cmax + 1, -1, dtype=np.int32)
    for i in range(C):
        class_to_idx[class_values[i]] = i

    conditional_entropy = 0.0
    background_cond_entropy = 0.0
    n = Y.size

    hist = np.zeros(C, dtype=np.uint32)
    hist_spoofed = np.zeros(C, dtype=np.uint32)

    for gi in prange(f_values.size):
        group_size = f_counts[gi]
        if group_size <= 1:
            continue

        start = group_starts[gi]
        end = start + group_size

        for c in range(C):
            hist[c] = 0
            if cardinality_correction:
                hist_spoofed[c] = 0

        for pidx in range(start, end):
            original_idx = positions[pidx]
            y_val = Y[original_idx]
            class_idx = class_to_idx[y_val]
            hist[class_idx] += 1

        if cardinality_correction:
            shift = group_size
            for pidx in range(start, end):
                original_idx = positions[pidx]
                spoofed_idx = (original_idx + shift) % n
                y_val_spoofed = Y[spoofed_idx]
                class_idx_spoofed = class_to_idx[y_val_spoofed]
                hist_spoofed[class_idx_spoofed] += 1

        initial_prob = group_size / all_events
        conditional_entropy += compute_conditional_entropy(initial_prob, group_size, hist)
        if cardinality_correction:
            background_cond_entropy += compute_conditional_entropy(initial_prob, group_size, hist_spoofed)

    if not cardinality_correction:
        return full_entropy - conditional_entropy
    else:
        return -conditional_entropy + background_cond_entropy


@njit(
    'Tuple((int32[:], int32[:]))(int32[:], int32[:], float32, int32[:])',
    cache=True,
    fastmath=True
)
def stratified_subsampling(Y, X, approximation_factor, _f_values_X):
    """
    More efficient subsampling that avoids repeated np.where scans.
    """
    all_events = X.size
    final_space_size = int(approximation_factor * all_events)
    if _f_values_X.size == 0:
        return Y, X
    unique_samples_per_val = int(final_space_size / _f_values_X.size)
    if unique_samples_per_val == 0:
        return Y, X

    final_index_array = np.empty(final_space_size, dtype=np.int32)
    index_offset = 0

    for fval in _f_values_X:
        count_collected = 0
        for j in range(X.size):
            if X[j] == fval:
                if count_collected < unique_samples_per_val:
                    if index_offset < final_space_size:
                        final_index_array[index_offset] = j
                        index_offset += 1
                    count_collected += 1
                else:
                    break

    final_index_array = final_index_array[:index_offset]
    X_sub = X[final_index_array]
    Y_sub = Y[final_index_array]
    return Y_sub, X_sub


@njit(
    'float32(int32[:], int32[:], float32, b1)',
    cache=True,
    fastmath=True,
)
def mutual_info_estimator_numba_opt(
    Y, X, approximation_factor=1.0, cardinality_correction=False,
):
    """
    The heuristic is MI-numba-randomized, but the code for numba is structured so the execution is faster.
    Core estimator logic. This version uses the efficient grouped approach.
    """
    
    if X.size != Y.size:
        raise ValueError("Input arrays X and Y must have the same length.")
    if X.size == 0:
        raise ValueError("Input arrays cannot be empty.")
    
    all_events = X.size

    is_diagonal = True
    if X.size == Y.size:
        for i in range(X.size):
            if X[i] != Y[i]:
                is_diagonal = False
                break
    else:
        is_diagonal = False

    if is_diagonal:
        cardinality_correction = False

    if approximation_factor < 1.0:
        f_values_full, _ = numba_unique(X)
        Y, X = stratified_subsampling(Y, X, approximation_factor, f_values_full)
        all_events = X.size

    f_values, f_counts, group_starts, positions = build_groups(X)

    joint_entropy_core = compute_entropies_grouped(
        Y, all_events, f_values, f_counts, group_starts, positions, cardinality_correction
    )

    return approximation_factor * joint_entropy_core