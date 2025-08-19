from __future__ import annotations

from collections import Counter


class PrimitiveConstrainedCounter:
    """
    A memory-efficient implementation of the count min sketch algorithm with optimized hashing using Numba JIT.
    """

    def __init__(self, bound: int=(10**4) * 3):
        self.max_bound_thr = bound
        self.default_counter: Counter = Counter()

    def batch_add(self, lst):
        if len(self.default_counter) < self.max_bound_thr:
            # More efficient: update instead of creating new counter
            self.default_counter.update(lst)

    def add(self, val):
        if len(self.default_counter) < self.max_bound_thr:
            self.default_counter[val] += 1


if __name__ == '__main__':
    from collections import Counter

    depth = 8
    width = 2**22
    cms = PrimitiveConstrainedCounter()

    items = [1, 1, 2, 3, 3, 3, 4, 5, 2] * 10000
    cms.batch_add(items)  # Use the batch_add function

    print(Counter(items))  # Print the exact counts for comparison
