from __future__ import annotations

from collections import Counter

import numpy as np
from numba import njit
from numba import prange


@njit
def cms_hash(x, seed, width):
    x_hash = np.uint32(hash(x) & 0xFFFFFFFF)
    return (x_hash + seed) % width

class CountMinSketch:
    """
    A memory-efficient implementation of the count min sketch algorithm with optimized hashing using Numba JIT.
    """

    def __init__(self, depth=6, width=2**15, M=None):
        self.depth = depth
        self.width = width
        self.hash_seeds = np.array(np.random.randint(low=0, high=2**31 - 1, size=depth), dtype=np.uint32)
        self.M = np.zeros((depth, width), dtype=np.int32) if M is None else M

    @staticmethod
    @njit
    def _add(M, x, depth, width, hash_seeds, delta=1):
        for i in prange(depth):
            location = cms_hash(x, hash_seeds[i], width)
            M[i, location] += delta

    def add(self, x, delta=1):
        CountMinSketch._add(self.M, x, self.depth, self.width, self.hash_seeds, delta)

    def batch_add(self, lst, delta=1):
        for x in lst:
            self.add(x, delta)

    def query(self, x):
        return min(self.M[i][cms_hash(x, self.hash_seeds[i], self.width)] for i in range(self.depth))

    def get_matrix(self):
        return self.M


if __name__ == '__main__':
    from collections import Counter

    depth = 8
    width = 2**22
    cms = CountMinSketch(depth, width)

    items = [1, 1, 2, 3, 3, 3, 4, 5, 2] * 1000
    cms.batch_add(items)  # Use the batch_add function

    print(cms.query(3))  # Query for frequency estimates
    print(cms.query(1))
    print(cms.query(2))
    print(cms.query(4))
    print(cms.query(5))

    print(Counter(items))  # Print the exact counts for comparison
