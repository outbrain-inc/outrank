from __future__ import annotations

import numpy as np

np.random.seed(123)
max_size = 10**6


def max_pair_coverage(array1: np.array, array2: np.array) -> float:
    def hash_pair(el1, el2):
        return el1 * 17 - el2

    counts = np.zeros(max_size, dtype=np.int32)
    tot_len = len(array1)
    for i in range(tot_len):
        identifier = hash_pair(array1[i], array2[i])
        counts[identifier % max_size] += 1

    return np.max(counts) / tot_len


if __name__ == '__main__':

    array1 = np.array([1,1,2,3,1,1,1,5] * 1000)
    array2 = np.array([0,0,5,5,3,0,0,0] * 1000)
    coverage = max_pair_coverage(array1, array2)
    assert coverage == 0.5