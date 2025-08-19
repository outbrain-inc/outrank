"""
This module implements probabilistic data structure which is able to calculate the cardinality of large multisets in a single pass using little auxiliary memory
"""
from __future__ import annotations

import numpy as np
import xxhash


class HyperLogLogWCache:
    def __init__(self, error_rate=0.005):
        # int(np.ceil(np.log2((1.04 / error_rate) ** 2)))
        self.p = 19
        self.m = 1 << self.p
        self.warmup_set = set()
        self.warmup_size = int(self.m / 2)
        self.width = 64 - self.p
        self.hll_flag = False

    def _hasher_update(self, value):
        self.hasher = xxhash.xxh32(seed=self.p)
        if isinstance(value, str):
            value = value.encode('utf-8')
            self.hasher.update(bytes(value))
        else:
            self.hasher.update(bytes(value))

        x = self.hasher.intdigest()
        j = x & (self.m - 1)
        w = x >> self.p

        rho = self.width - w.bit_length()
        self.M[j] = max(self.M[j], rho)

    def add(self, value):
        if len(self.warmup_set) < self.warmup_size and not self.hll_flag:
            self.warmup_set.add(value)
        elif not self.hll_flag:
            # Initialize HyperLogLog when warmup is complete
            self.M = np.zeros(self.m)
            for element in self.warmup_set:
                self._hasher_update(element)
            self.warmup_set = set()  # Clear warmup set and convert to set for consistency
            self.hll_flag = True
        else:
            self._hasher_update(value)

    def __len__(self):
        if self.hll_flag:
            # Optimize: calculate zero count once
            zero_count = len(np.where(self.M == 0)[0])
            if zero_count > 0:
                basis = np.ceil(self.m * np.log(self.m / zero_count))
                if basis != np.inf:
                    return int(basis) - 1
            return 2**self.p
        else:
            return len(self.warmup_set)


if __name__ == '__main__':
    import random
    import string


    def get_random_string(length):
        # choose from all lowercase letter
        letters = string.ascii_lowercase
        result_str = ''.join(random.choice(letters) for i in range(length))
        return result_str

    # results_df = []
    # num_vals = 100000
    # nbits = 16
    # for _ in range(3):
    #     for j in tqdm.tqdm(range(1000000, 10000000, 1000)):
    #         ground = list(set(np.random.randint(0, j, num_vals).tolist()))
    #         ground = ground + [
    #             get_random_string(random.randint(1, 15)) for k in range(j)
    #         ]

    #         start_time = time.time()
    #         GLOBAL_CARDINALITY_STORAGE = {}
    #         GLOBAL_CARDINALITY_STORAGE[1] = HyperLogLogWCache(0.005)

    #         for j in ground:
    #             GLOBAL_CARDINALITY_STORAGE[1].add(j)

    #         size1 = asizeof.asizeof(GLOBAL_CARDINALITY_STORAGE)
    #         error1 = 100 * \
    #             (1 - len(GLOBAL_CARDINALITY_STORAGE[1]) / len(set(ground)))
    #         end_time = time.time()
    #         tp1 = end_time - start_time

    #         import hyperloglog

    #         start_time = time.time()
    #         GLOBAL_CARDINALITY_STORAGE = {}
    #         GLOBAL_CARDINALITY_STORAGE[1] = hyperloglog.HyperLogLog(0.005)

    #         for j in ground:
    #             GLOBAL_CARDINALITY_STORAGE[1].add(j)
    #         size2 = asizeof.asizeof(GLOBAL_CARDINALITY_STORAGE)
    #         error2 = 100 * \
    #             (1 - len(GLOBAL_CARDINALITY_STORAGE[1]) / len(set(ground)))
    #         end_time = time.time()
    #         tp2 = end_time - start_time

    #         start_time = time.time()
    #         GLOBAL_CARDINALITY_STORAGE = set()

    #         for j in ground:
    #             GLOBAL_CARDINALITY_STORAGE.add(j)

    #         size3 = asizeof.asizeof(GLOBAL_CARDINALITY_STORAGE)
    #         error3 = 100 * \
    #             (1 - len(GLOBAL_CARDINALITY_STORAGE) / len(set(ground)))
    #         end_time = time.time()
    #         tp3 = end_time - start_time

    #         results_df.append(
    #             {
    #                 'num_samples': len(ground),
    #                 'time': tp3,
    #                 'algo': 'set',
    #                 'error': error3,
    #             },
    #         )
    #         results_df.append(
    #             {
    #                 'num_samples': len(ground),
    #                 'time': tp2,
    #                 'algo': 'default',
    #                 'error': error2,
    #             },
    #         )
    #         results_df.append(
    #             {
    #                 'num_samples': len(ground),
    #                 'time': tp1,
    #                 'algo': f'hllc ({nbits}, mixed)',
    #                 'error': error1,
    #             },
    #         )

    # out_df = pd.DataFrame(results_df)
    # out_df.to_csv('backup.csv')
    # print(out_df)
    # print(out_df.groupby('algo').mean())
    # sns.lineplot(
    #     x=out_df.num_samples, y=out_df.error,
    #     hue=out_df.algo, alpha=0.5,
    # )
    # plt.tight_layout()
    # plt.ylabel('Num. of unique values in data')
    # plt.ylabel('Abs error')
    # plt.savefig('linep.pdf')
    # plt.clf()
    # plt.cla()

    # sns.lineplot(
    #     x=out_df.num_samples.astype(
    #         float,
    #     ), y=out_df.time, hue=out_df.algo,
    # )
    # plt.tight_layout()
    # plt.ylabel('Time (s)')
    # plt.savefig('barp.pdf')
    # plt.clf()
    # plt.cla()
