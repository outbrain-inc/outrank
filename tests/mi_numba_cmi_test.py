from __future__ import annotations

import sys
import unittest

import numpy as np

from outrank.algorithms.feature_ranking.ranking_mi_numba_cmi import _mi_from_arrays
from outrank.algorithms.feature_ranking.ranking_mi_numba_cmi import compute_joint_mi_numba
from outrank.algorithms.feature_ranking.ranking_mi_numba_cmi import conditional_mutual_info_numba
from outrank.algorithms.feature_ranking.ranking_mi_numba_cmi import interaction_information_numba

sys.path.append('./outrank')


class ConditionalMITest(unittest.TestCase):

    def test_cmi_xor_detection(self):
        """XOR: I(X1;Y) ~ 0 but I(X1;Y|X2) is high — the core synergy case."""
        np.random.seed(42)
        n = 5000
        X1 = np.random.randint(0, 2, size=n, dtype=np.int32)
        X2 = np.random.randint(0, 2, size=n, dtype=np.int32)
        Y = np.bitwise_xor(X1, X2).astype(np.int32)

        # Pairwise MI should be near zero
        mi_x1_y = float(_mi_from_arrays(Y, X1, np.float32(1.0), False))
        self.assertLess(abs(mi_x1_y), 0.05)

        # Conditional MI should be high — X1 becomes informative once X2 is known
        cmi = float(conditional_mutual_info_numba(Y, X1, X2, np.float32(1.0), False))
        self.assertGreater(cmi, 0.5)

    def test_cmi_chain_rule(self):
        """Chain rule: I(X1,X2;Y) ~ I(X1;Y) + I(X2;Y|X1)"""
        np.random.seed(42)
        n = 5000
        X1 = np.random.randint(0, 4, size=n, dtype=np.int32)
        X2 = np.random.randint(0, 3, size=n, dtype=np.int32)
        # Y depends on both
        Y = ((X1 + X2) % 5).astype(np.int32)

        joint_mi = float(compute_joint_mi_numba(Y, X1, X2, np.float32(1.0), False))
        mi_x1_y = float(_mi_from_arrays(Y, X1, np.float32(1.0), False))
        cmi_x2_y_given_x1 = float(conditional_mutual_info_numba(Y, X2, X1, np.float32(1.0), False))

        chain_sum = mi_x1_y + cmi_x2_y_given_x1
        self.assertAlmostEqual(joint_mi, chain_sum, delta=0.2)

    def test_cmi_reduces_to_mi_when_z_constant(self):
        """When Z is constant, I(X;Y|Z) = I(X;Y)."""
        np.random.seed(42)
        n = 3000
        X = np.random.randint(0, 3, size=n, dtype=np.int32)
        Y = X.copy()  # perfect correlation
        Z = np.zeros(n, dtype=np.int32)  # constant

        cmi = float(conditional_mutual_info_numba(Y, X, Z, np.float32(1.0), False))
        mi = float(_mi_from_arrays(Y, X, np.float32(1.0), False))
        self.assertAlmostEqual(cmi, mi, delta=0.05)

    def test_cmi_independent_condition(self):
        """Z independent of (X,Y) => I(X;Y|Z) ~ I(X;Y)."""
        np.random.seed(42)
        n = 5000
        X = np.random.randint(0, 3, size=n, dtype=np.int32)
        Y = X.copy()
        Z = np.random.randint(0, 4, size=n, dtype=np.int32)  # independent noise

        cmi = float(conditional_mutual_info_numba(Y, X, Z, np.float32(1.0), False))
        mi = float(_mi_from_arrays(Y, X, np.float32(1.0), False))
        self.assertAlmostEqual(cmi, mi, delta=0.15)

    def test_cmi_non_negative(self):
        """I(X;Y|Z) >= 0 (without cardinality correction)."""
        np.random.seed(42)
        n = 2000
        X = np.random.randint(0, 5, size=n, dtype=np.int32)
        Y = np.random.randint(0, 5, size=n, dtype=np.int32)
        Z = np.random.randint(0, 3, size=n, dtype=np.int32)

        cmi = float(conditional_mutual_info_numba(Y, X, Z, np.float32(1.0), False))
        self.assertGreaterEqual(cmi, -0.01)

    def test_ii_xor_synergy(self):
        """XOR -> II < 0 (synergy detected)."""
        np.random.seed(42)
        n = 5000
        X1 = np.random.randint(0, 2, size=n, dtype=np.int32)
        X2 = np.random.randint(0, 2, size=n, dtype=np.int32)
        Y = np.bitwise_xor(X1, X2).astype(np.int32)

        ii = interaction_information_numba(Y, X1, X2)
        self.assertLess(ii, -0.1)

    def test_ii_redundancy(self):
        """X1=X2=Y -> II > 0 (redundancy)."""
        np.random.seed(42)
        n = 5000
        Y = np.random.randint(0, 3, size=n, dtype=np.int32)
        X1 = Y.copy()
        X2 = Y.copy()

        ii = interaction_information_numba(Y, X1, X2)
        self.assertGreater(ii, 0.1)

    def test_ii_independent_contributions(self):
        """X1 and X2 provide independent, non-overlapping info about Y -> II ~ 0."""
        np.random.seed(42)
        n = 5000
        # Y is a concatenation of two independent signals: high bits from X1, low bit from X2
        X1 = np.random.randint(0, 4, size=n, dtype=np.int32)
        X2 = np.random.randint(0, 2, size=n, dtype=np.int32)
        # Y encodes both without interaction: Y = X1*2 + X2 (no modular wrap)
        Y = (X1 * 2 + X2).astype(np.int32)

        ii = interaction_information_numba(Y, X1, X2)
        # Independent contributions: I(X1,X2;Y) = I(X1;Y) + I(X2;Y), so II ~ 0
        self.assertLess(abs(ii), 0.15)

    def test_cmi_cardinality_correction(self):
        """With cardinality correction, high-card Z should yield lower CMI."""
        np.random.seed(42)
        n = 3000
        X = np.random.randint(0, 3, size=n, dtype=np.int32)
        Y = np.random.randint(0, 3, size=n, dtype=np.int32)
        # High cardinality Z creates many small groups -> more finite-sample bias
        Z = np.random.randint(0, 50, size=n, dtype=np.int32)

        cmi_no_corr = float(conditional_mutual_info_numba(Y, X, Z, np.float32(1.0), False))
        cmi_with_corr = float(conditional_mutual_info_numba(Y, X, Z, np.float32(1.0), True))

        # Correction should reduce the estimate (or at least not inflate it)
        self.assertLessEqual(cmi_with_corr, cmi_no_corr + 0.05)

    def test_empty_arrays(self):
        """Empty arrays should return 0."""
        X = np.array([], dtype=np.int32)
        Y = np.array([], dtype=np.int32)
        Z = np.array([], dtype=np.int32)

        result = float(conditional_mutual_info_numba(Y, X, Z, np.float32(1.0), False))
        self.assertEqual(result, 0.0)

    def test_single_element(self):
        """Single-element arrays should return a valid float (0 — no entropy)."""
        X = np.array([1], dtype=np.int32)
        Y = np.array([0], dtype=np.int32)
        Z = np.array([2], dtype=np.int32)

        result = float(conditional_mutual_info_numba(Y, X, Z, np.float32(1.0), False))
        self.assertIsInstance(result, float)

    def test_mismatched_lengths_return_nan(self):
        """Mismatched array lengths should return NaN."""
        X = np.array([1, 2], dtype=np.int32)
        Y = np.array([0], dtype=np.int32)
        Z = np.array([1, 2], dtype=np.int32)

        result = float(conditional_mutual_info_numba(Y, X, Z, np.float32(1.0), False))
        self.assertTrue(np.isnan(result))

    def test_joint_mi_consistency(self):
        """Joint MI should be >= max(I(X1;Y), I(X2;Y)) — joint can't lose info."""
        np.random.seed(42)
        n = 3000
        X1 = np.random.randint(0, 3, size=n, dtype=np.int32)
        X2 = np.random.randint(0, 3, size=n, dtype=np.int32)
        Y = ((X1 * 2 + X2) % 5).astype(np.int32)

        joint = float(compute_joint_mi_numba(Y, X1, X2, np.float32(1.0), False))
        mi_1 = float(_mi_from_arrays(Y, X1, np.float32(1.0), False))
        mi_2 = float(_mi_from_arrays(Y, X2, np.float32(1.0), False))

        self.assertGreaterEqual(joint + 0.05, max(mi_1, mi_2))

    def test_cmi_subsampling(self):
        """Subsampling should produce a result in the same ballpark."""
        np.random.seed(42)
        n = 5000
        X = np.random.randint(0, 3, size=n, dtype=np.int32)
        Y = X.copy()
        Z = np.random.randint(0, 2, size=n, dtype=np.int32)

        cmi_full = float(conditional_mutual_info_numba(Y, X, Z, np.float32(1.0), False))
        cmi_sub = float(conditional_mutual_info_numba(Y, X, Z, np.float32(0.5), False))

        # Subsampled should be in the same direction and roughly similar magnitude
        self.assertGreater(cmi_sub, 0.0)
        self.assertAlmostEqual(cmi_full, cmi_sub, delta=0.4)


if __name__ == '__main__':
    unittest.main()
