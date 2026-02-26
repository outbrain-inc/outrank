from __future__ import annotations

import numpy as np
from numba import njit
from numba import prange


# --- Copied from ranking_mi_numba_opt.py ---
# Numba cross-module @njit calls with cache=True fail under certain compilation
# orderings. Copying these ~60 lines is the pragmatic solution. These functions
# must stay in sync with the originals in ranking_mi_numba_opt.py.

@njit('Tuple((int32[:], int32[:]))(int32[:])', cache=True, fastmath=True)
def _numba_unique(a):
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
def _compute_conditional_entropy(initial_prob, group_size, class_counts):
    ce = 0.0
    inv_group_size = 1.0 / group_size
    for count in class_counts:
        if count > 0:
            conditional_prob = count * inv_group_size
            ce -= initial_prob * conditional_prob * np.log(conditional_prob)
    return ce


@njit('Tuple((int32[:], int32[:], int32[:], int32[:]))(int32[:])', cache=True, fastmath=True)
def _build_groups(X):
    f_values, f_counts = _numba_unique(X)
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
def _compute_entropies_grouped(
    Y, all_events,
    f_values, f_counts, group_starts, positions,
    cardinality_correction,
):
    class_values, class_counts = _numba_unique(Y)
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
        conditional_entropy += _compute_conditional_entropy(initial_prob, group_size, hist)
        if cardinality_correction:
            background_cond_entropy += _compute_conditional_entropy(initial_prob, group_size, hist_spoofed)

    if not cardinality_correction:
        return full_entropy - conditional_entropy
    else:
        return -conditional_entropy + background_cond_entropy


# --- End copied section ---


@njit(
    'float32(int32[:], int32[:], float32, b1)',
    cache=True,
    fastmath=True,
)
def _mi_from_arrays(Y, X, approximation_factor, cardinality_correction):
    """Compute MI(X;Y) from flat int32 arrays — internal helper replicating
    the core logic of mutual_info_estimator_numba_opt without the subsampling
    path (callers are responsible for pre-subsample)."""
    all_events = np.int32(X.size)
    f_values, f_counts, group_starts, positions = _build_groups(X)
    return _compute_entropies_grouped(
        Y, all_events, f_values, f_counts, group_starts, positions,
        cardinality_correction,
    )


@njit(
    'float32(int32[:], int32[:], int32[:], int32, int32, int32, int32)',
    cache=True,
    fastmath=True,
)
def _compute_cmi_contingency(Y, X, Z, all_events, dx, dy, dz):
    """I(X;Y|Z) via contingency tables — single-pass count accumulation.

    ~2x faster than per-Z-group approach by building flat count arrays
    count[x,y,z], count[x,z], count[y,z], count[z] in one pass over data,
    then iterating non-zero entries to compute the CMI sum directly.
    Avoids repeated _build_groups calls and per-group sub-array extraction.
    Only used when cardinality_correction is False.
    """
    dy_dz = dy * dz

    count_xyz = np.zeros(dx * dy_dz, dtype=np.int32)
    count_xz = np.zeros(dx * dz, dtype=np.int32)
    count_yz = np.zeros(dy_dz, dtype=np.int32)
    count_z = np.zeros(dz, dtype=np.int32)

    for i in range(all_events):
        x, y, z = X[i], Y[i], Z[i]
        count_xyz[x * dy_dz + y * dz + z] += 1
        count_xz[x * dz + z] += 1
        count_yz[y * dz + z] += 1
        count_z[z] += 1

    inv_n = np.float32(1.0) / np.float32(all_events)
    cmi = np.float32(0.0)

    for x in range(dx):
        x_off_xyz = x * dy_dz
        x_off_xz = x * dz
        for y in range(dy):
            y_off = y * dz
            for z in range(dz):
                nxyz = count_xyz[x_off_xyz + y_off + z]
                if nxyz == 0:
                    continue
                nxz = count_xz[x_off_xz + z]
                nyz = count_yz[y_off + z]
                nz = count_z[z]
                # p(x,y,z) * log( p(x,y|z) / (p(x|z) * p(y|z)) )
                cmi += np.float32(nxyz) * inv_n * np.log(
                    (np.float32(nxyz) * np.float32(nz))
                    / (np.float32(nxz) * np.float32(nyz)),
                )

    return cmi


@njit(
    'float32(int32[:], int32[:], int32[:], int32, b1)',
    cache=True,
    fastmath=True,
)
def _compute_cmi_core(Y, X, Z, all_events, cardinality_correction):
    """I(X;Y|Z) — dispatches to contingency table (fast) or per-Z-group (fallback).

    Fast path: single-pass contingency tables when cardinality_correction is off
    and the joint cardinality product is manageable (<=2M cells, ~8MB).
    Fallback: per-Z-group partitioning with _build_groups for cardinality
    correction or high cardinality.
    """
    # --- Fast path: contingency tables (no cardinality correction) ---
    if not cardinality_correction:
        dx = np.int32(0)
        dy = np.int32(0)
        dz = np.int32(0)
        for i in range(all_events):
            if X[i] > dx:
                dx = X[i]
            if Y[i] > dy:
                dy = Y[i]
            if Z[i] > dz:
                dz = Z[i]
        dx += np.int32(1)
        dy += np.int32(1)
        dz += np.int32(1)
        if np.int64(dx) * np.int64(dy) * np.int64(dz) <= np.int64(2_000_000):
            return _compute_cmi_contingency(Y, X, Z, all_events, dx, dy, dz)

    # --- Fallback: per-Z-group partitioning ---
    z_values, z_counts = _numba_unique(Z)
    n_z = z_values.size

    # Build index mapping for Z so we can extract subsets without repeated scans
    zmax = np.int32(0)
    for i in range(n_z):
        if z_values[i] > zmax:
            zmax = z_values[i]
    z_to_idx = np.full(zmax + 1, -1, dtype=np.int32)
    for i in range(n_z):
        z_to_idx[z_values[i]] = i

    # Pre-compute group start positions for Z
    z_starts = np.zeros(n_z, dtype=np.int32)
    run = np.int32(0)
    for i in range(n_z):
        z_starts[i] = run
        run += z_counts[i]

    z_positions = np.empty(all_events, dtype=np.int32)
    z_cursors = z_starts.copy()
    for i in range(all_events):
        zi = Z[i]
        gi = z_to_idx[zi]
        pos = z_cursors[gi]
        z_positions[pos] = i
        z_cursors[gi] = pos + 1

    # Allocate working arrays — sized to largest Z-group to avoid per-loop alloc
    max_group = np.int32(0)
    for i in range(n_z):
        if z_counts[i] > max_group:
            max_group = z_counts[i]

    X_sub = np.empty(max_group, dtype=np.int32)
    Y_sub = np.empty(max_group, dtype=np.int32)

    cmi = np.float32(0.0)

    for zi in range(n_z):
        gz = z_counts[zi]
        if gz <= 1:
            continue

        start = z_starts[zi]
        for j in range(gz):
            idx = z_positions[start + j]
            X_sub[j] = X[idx]
            Y_sub[j] = Y[idx]

        X_slice = X_sub[:gz]
        Y_slice = Y_sub[:gz]

        # MI(X;Y) for this Z-group
        f_values_x, f_counts_x, group_starts_x, positions_x = _build_groups(X_slice)
        mi_z = _compute_entropies_grouped(
            Y_slice, gz,
            f_values_x, f_counts_x, group_starts_x, positions_x,
            cardinality_correction,
        )

        # Weight by P(Z=z)
        p_z = np.float32(gz) / np.float32(all_events)
        cmi += p_z * mi_z

    return cmi


@njit(
    'float32(int32[:], int32[:], int32[:], float32, b1)',
    cache=True,
    fastmath=True,
)
def conditional_mutual_info_numba(Y, X, Z, approximation_factor, cardinality_correction):
    """I(X;Y|Z) — public entry point.

    Validates inputs, applies optional subsampling, then delegates to the
    core CMI computation.
    """
    if X.size != Y.size or X.size != Z.size:
        # Numba doesn't support raising with message; return NaN sentinel
        return np.float32(np.nan)
    if X.size == 0:
        return np.float32(0.0)

    all_events = np.int32(X.size)

    # Subsampling: uniform random (not stratified — three-variable stratification
    # is combinatorially expensive and not worth it for CMI where the outer
    # Z-conditioning already partitions the data).
    if approximation_factor < 1.0:
        target_n = np.int32(approximation_factor * all_events)
        if target_n < 2:
            target_n = np.int32(2)
        # Deterministic stride-based subsampling for Numba compatibility
        step = all_events // target_n
        if step < 1:
            step = np.int32(1)
        actual_n = (all_events + step - 1) // step
        X_s = np.empty(actual_n, dtype=np.int32)
        Y_s = np.empty(actual_n, dtype=np.int32)
        Z_s = np.empty(actual_n, dtype=np.int32)
        idx = np.int32(0)
        for i in range(0, all_events, step):
            X_s[idx] = X[i]
            Y_s[idx] = Y[i]
            Z_s[idx] = Z[i]
            idx += 1
        X_s = X_s[:idx]
        Y_s = Y_s[:idx]
        Z_s = Z_s[:idx]
        return _compute_cmi_core(Y_s, X_s, Z_s, idx, cardinality_correction)

    return _compute_cmi_core(Y, X, Z, all_events, cardinality_correction)


@njit(
    'float32(int32[:], int32[:], int32[:], float32, b1)',
    cache=True,
    fastmath=True,
)
def compute_joint_mi_numba(Y, X1, X2, approximation_factor, cardinality_correction):
    """I(X1,X2; Y) using integer-encoded joint feature.

    Encodes the joint (X1,X2) as a single integer: joint = X1 * (max(X2)+1) + X2.
    Pure integer arithmetic, no string hashing.
    """
    n = X1.size
    if n == 0:
        return np.float32(0.0)

    max_x2 = np.int32(0)
    for i in range(n):
        if X2[i] > max_x2:
            max_x2 = X2[i]
    base = max_x2 + np.int32(1)

    joint = np.empty(n, dtype=np.int32)
    for i in range(n):
        joint[i] = X1[i] * base + X2[i]

    return _mi_from_arrays(Y, joint, approximation_factor, cardinality_correction)


def interaction_information_numba(Y, X1, X2, approximation_factor=np.float32(1.0), cardinality_correction=False):
    """II(X1, X2; Y) = I(X1;Y) + I(X2;Y) - I({X1,X2};Y)

    Jakulin & Bratko convention:
    Negative = synergy (XOR-like: features are jointly informative but
    individually uninformative).
    Positive = redundancy (features carry overlapping information about Y).
    ~0 = independent contributions.

    This is a pure Python function (not @njit) because it orchestrates
    multiple Numba-compiled calls.
    """
    Y_i = Y.astype(np.int32)
    X1_i = X1.astype(np.int32)
    X2_i = X2.astype(np.int32)
    af = np.float32(approximation_factor)

    mi_x1_y = float(_mi_from_arrays(Y_i, X1_i, af, cardinality_correction))
    mi_x2_y = float(_mi_from_arrays(Y_i, X2_i, af, cardinality_correction))
    joint_mi = float(compute_joint_mi_numba(Y_i, X1_i, X2_i, af, cardinality_correction))

    return mi_x1_y + mi_x2_y - joint_mi
