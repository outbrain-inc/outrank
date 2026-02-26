from __future__ import annotations

import numpy as np
from numba import njit
from numba import prange

np.random.seed(123)


@njit('Tuple((int32[:], int32[:]))(int32[:])', cache=True, fastmath=True, boundscheck=False)
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


@njit('float32(float32, int32, uint32[:])', cache=True, fastmath=True, boundscheck=False)
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


@njit('Tuple((int32[:], int32[:], int32[:], int32[:]))(int32[:])', cache=True, fastmath=True, boundscheck=False)
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
    boundscheck=False,
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
        # CC estimates can go slightly negative due to finite-sample noise;
        # MI is non-negative by definition, so clamp to zero.
        result = -conditional_entropy + background_cond_entropy
        if result < np.float32(0.0):
            return np.float32(0.0)
        return result


@njit(
    'float32(int32[:], int32[:], int32, int32, int32)',
    cache=True,
    fastmath=True,
    boundscheck=False,
)
def _compute_mi_contingency(Y, X, all_events, dx, dy):
    """MI(X;Y) via contingency table — single-pass count accumulation.

    Builds flat count arrays count[x,y], count[x], count[y] in one pass,
    then computes MI = sum p(x,y) * log(p(x,y) / (p(x)*p(y))) over
    non-zero entries. ~2x faster than grouped approach for moderate
    cardinality because it avoids build_groups overhead and per-group
    histogram clearing.

    Only used when cardinality_correction is False and dx*dy <= 2M.
    """
    count_xy = np.zeros(dx * dy, dtype=np.int32)
    count_x = np.zeros(dx, dtype=np.int32)
    count_y = np.zeros(dy, dtype=np.int32)

    for i in range(all_events):
        x, y = X[i], Y[i]
        count_xy[x * dy + y] += 1
        count_x[x] += 1
        count_y[y] += 1

    inv_n = np.float32(1.0) / np.float32(all_events)
    mi = np.float32(0.0)

    # Dense iteration with branch-skip for zero entries. For the no-CC path
    # (single pass), this is already optimal — sparse collection overhead
    # negates any benefit. Sparse is only worthwhile for the CC path which
    # iterates the table twice (see _compute_mi_contingency_cc).
    for x in range(dx):
        nx = count_x[x]
        if nx == 0:
            continue
        x_off = x * dy
        for y in range(dy):
            nxy = count_xy[x_off + y]
            if nxy == 0:
                continue
            ny = count_y[y]
            mi += np.float32(nxy) * inv_n * np.log(
                (np.float32(nxy) * np.float32(all_events))
                / (np.float32(nx) * np.float32(ny)),
            )

    return mi


@njit(
    'Tuple((float32, float32))(int32[:], int32[:], int32[:], int32[:], int32, int32, float32)',
    cache=True,
    fastmath=True,
    boundscheck=False,
)
def _cc_sparse_entropies(count_xy, count_xy_spoofed, count_x, count_y, dx, dy, inv_n):
    """Sparse iteration for CC path — extracted to keep _compute_mi_contingency_cc
    small so LLVM doesn't spill registers on the dense fast path."""
    table_size = dx * dy
    cond_entropy = np.float32(0.0)
    bg_cond_entropy = np.float32(0.0)

    # Real table: collect non-zero entries
    nnz_real = np.int32(0)
    for k in range(table_size):
        if count_xy[k] > 0:
            nnz_real += 1
    nz_x_r = np.empty(nnz_real, dtype=np.int32)
    nz_nxy_r = np.empty(nnz_real, dtype=np.int32)
    idx = np.int32(0)
    for x in range(dx):
        x_off = x * dy
        for y in range(dy):
            c = count_xy[x_off + y]
            if c > 0:
                nz_x_r[idx] = x
                nz_nxy_r[idx] = c
                idx += 1

    for k in range(nnz_real):
        nxy = nz_nxy_r[k]
        nx = count_x[nz_x_r[k]]
        if nx <= 1:
            continue
        px = np.float32(nx) * inv_n
        inv_nx = np.float32(1.0) / np.float32(nx)
        p_y_given_x = np.float32(nxy) * inv_nx
        cond_entropy -= px * p_y_given_x * np.log(p_y_given_x)

    # Spoofed table
    nnz_spoof = np.int32(0)
    for k in range(table_size):
        if count_xy_spoofed[k] > 0:
            nnz_spoof += 1
    nz_x_s = np.empty(nnz_spoof, dtype=np.int32)
    nz_nxy_s = np.empty(nnz_spoof, dtype=np.int32)
    idx = np.int32(0)
    for x in range(dx):
        x_off = x * dy
        for y in range(dy):
            c = count_xy_spoofed[x_off + y]
            if c > 0:
                nz_x_s[idx] = x
                nz_nxy_s[idx] = c
                idx += 1

    for k in range(nnz_spoof):
        nxy_s = nz_nxy_s[k]
        nx = count_x[nz_x_s[k]]
        if nx <= 1:
            continue
        px = np.float32(nx) * inv_n
        inv_nx = np.float32(1.0) / np.float32(nx)
        p_y_given_x_s = np.float32(nxy_s) * inv_nx
        bg_cond_entropy -= px * p_y_given_x_s * np.log(p_y_given_x_s)

    return cond_entropy, bg_cond_entropy


@njit(
    'float32(int32[:], int32[:], int32, int32, int32)',
    cache=True,
    fastmath=True,
    boundscheck=False,
)
def _compute_mi_contingency_cc(Y, X, all_events, dx, dy):
    """MI(X;Y) with cardinality correction via contingency table.

    Computes H_bg(Y|X) - H(Y|X) where H_bg uses spoofed Y assignment:
    Y_spoofed[i] = Y[(i + count_x[X[i]]) % n]. Sparse iteration extracted
    to _cc_sparse_entropies to avoid LLVM register spill on the dense path.
    """
    n = all_events

    # Pass 1: build real contingency table and marginals
    count_xy = np.zeros(dx * dy, dtype=np.int32)
    count_x = np.zeros(dx, dtype=np.int32)
    count_y = np.zeros(dy, dtype=np.int32)

    for i in range(n):
        x, y = X[i], Y[i]
        count_xy[x * dy + y] += 1
        count_x[x] += 1
        count_y[y] += 1

    # Pass 2: build spoofed contingency table
    count_xy_spoofed = np.zeros(dx * dy, dtype=np.int32)

    for i in range(n):
        x = X[i]
        spoofed_idx = (i + count_x[x]) % n
        y_spoofed = Y[spoofed_idx]
        count_xy_spoofed[x * dy + y_spoofed] += 1

    inv_n = np.float32(1.0) / np.float32(n)
    table_size = dx * dy

    if table_size > 4 * n:
        cond_entropy, bg_cond_entropy = _cc_sparse_entropies(
            count_xy, count_xy_spoofed, count_x, count_y, dx, dy, inv_n,
        )
    else:
        cond_entropy = np.float32(0.0)
        bg_cond_entropy = np.float32(0.0)
        for x in range(dx):
            nx = count_x[x]
            if nx <= 1:
                continue
            px = np.float32(nx) * inv_n
            inv_nx = np.float32(1.0) / np.float32(nx)
            x_off = x * dy
            for y in range(dy):
                nxy = count_xy[x_off + y]
                if nxy > 0:
                    p_y_given_x = np.float32(nxy) * inv_nx
                    cond_entropy -= px * p_y_given_x * np.log(p_y_given_x)

        for x in range(dx):
            nx = count_x[x]
            if nx <= 1:
                continue
            px = np.float32(nx) * inv_n
            inv_nx = np.float32(1.0) / np.float32(nx)
            x_off = x * dy
            for y in range(dy):
                nxy_s = count_xy_spoofed[x_off + y]
                if nxy_s > 0:
                    p_y_given_x_s = np.float32(nxy_s) * inv_nx
                    bg_cond_entropy -= px * p_y_given_x_s * np.log(p_y_given_x_s)

    # CC estimates can go slightly negative; MI is non-negative by definition.
    result = -cond_entropy + bg_cond_entropy
    if result < np.float32(0.0):
        return np.float32(0.0)
    return result


@njit(
    'Tuple((int32[:], int32[:]))(int32[:], int32[:], float32, int32[:], int32[:], int32[:], int32[:])',
    cache=True,
    fastmath=True,
    boundscheck=False,
)
def stratified_subsampling(Y, X, approximation_factor, f_values, f_counts, group_starts, positions):
    """O(n) stratified subsampling using pre-built group structure.

    Takes at most `samples_per_val` original indices from each group via
    the positions array (already partitioned by build_groups), avoiding
    any per-value linear scan of X.
    """
    all_events = X.size
    final_space_size = int(approximation_factor * all_events)
    n_groups = f_values.size
    if n_groups == 0:
        return Y, X
    samples_per_val = int(final_space_size / n_groups)
    if samples_per_val == 0:
        return Y, X

    final_index_array = np.empty(final_space_size, dtype=np.int32)
    offset = 0

    for gi in range(n_groups):
        start = group_starts[gi]
        take = min(samples_per_val, f_counts[gi])
        if offset + take > final_space_size:
            take = final_space_size - offset
        for j in range(take):
            final_index_array[offset] = positions[start + j]
            offset += 1
        if offset >= final_space_size:
            break

    final_index_array = final_index_array[:offset]
    return Y[final_index_array], X[final_index_array]


@njit(
    'float32(int32[:], int32[:], float32, b1)',
    cache=True,
    fastmath=True,
    boundscheck=False,
)
def mutual_info_estimator_numba_opt(
    Y, X, approximation_factor=1.0, cardinality_correction=False,
):
    """
    The heuristic is MI-numba-randomized, but the code for numba is structured so the execution is faster.
    Core estimator logic. This version uses the efficient grouped approach.
    """

    if X.size != Y.size:
        raise ValueError('Input arrays X and Y must have the same length.')
    if X.size == 0:
        raise ValueError('Input arrays cannot be empty.')

    all_events = X.size

    # Separate diagonal check (early exit) + max-value scan.
    # Fusing adds ~25% per-iteration overhead from the extra branch, which
    # hurts the common case (non-diagonal pairs where diagonal exits at i≈0).
    is_diagonal = True
    for i in range(all_events):
        if X[i] != Y[i]:
            is_diagonal = False
            break

    if is_diagonal:
        cardinality_correction = False

    # Fast path: when no subsampling needed, try contingency table dispatch
    # *before* paying the cost of build_groups (~100-220us).
    if approximation_factor >= 1.0:
        dx = np.int32(0)
        dy = np.int32(0)
        for i in range(all_events):
            if X[i] > dx:
                dx = X[i]
            if Y[i] > dy:
                dy = Y[i]
        dx += np.int32(1)
        dy += np.int32(1)
        if np.int64(dx) * np.int64(dy) <= np.int64(2_000_000):
            if not cardinality_correction:
                return _compute_mi_contingency(Y, X, all_events, dx, dy)
            else:
                return _compute_mi_contingency_cc(Y, X, all_events, dx, dy)

    # Slow path: subsampling or high-cardinality fallback — needs build_groups
    f_values, f_counts, group_starts, positions = build_groups(X)

    if approximation_factor < 1.0:
        Y, X = stratified_subsampling(Y, X, approximation_factor, f_values, f_counts, group_starts, positions)
        all_events = X.size
        # Rebuild groups on the subsampled data
        f_values, f_counts, group_starts, positions = build_groups(X)

    joint_entropy_core = compute_entropies_grouped(
        Y, all_events, f_values, f_counts, group_starts, positions, cardinality_correction,
    )

    return approximation_factor * joint_entropy_core
