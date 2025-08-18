from __future__ import annotations

import sys
import unittest

import numpy as np

from outrank.algorithms.feature_ranking.ranking_mi_numba import \
    mutual_info_estimator_numba

np.random.seed(123)
sys.path.append('./outrank')


class CompareStrategiesTest(unittest.TestCase):
    def test_mi_numba(self):
        a = np.random.random(10**6).reshape(-1).astype(np.int32)
        b = np.random.random(10**6).reshape(-1).astype(np.int32)
        final_score = mutual_info_estimator_numba(a, b, np.float32(1.0), False)
        self.assertEqual(final_score, 0.0)

    def test_mi_numba_random(self):
        a = np.array([1, 0, 0, 0, 1, 1, 1, 0], dtype=np.int32)
        b = np.random.random(8).reshape(-1).astype(np.int32)

        final_score = mutual_info_estimator_numba(a, b, np.float32(1.0), False)
        self.assertLess(final_score, 0.0)

    def test_mi_numba_mirror(self):
        a = np.array([1, 0, 0, 0, 1, 1, 1, 0], dtype=np.int32)
        b = np.array([1, 0, 0, 0, 1, 1, 1, 0], dtype=np.int32)
        final_score = mutual_info_estimator_numba(a, b, np.float32(1.0), False)
        self.assertGreater(final_score, 0.60)

    def test_mi_numba_longer_inputs(self):
        b = np.array([1, 0, 0, 0, 1, 1, 1, 0] * 10**5, dtype=np.int32)
        final_score = mutual_info_estimator_numba(b, b, np.float32(1.0), False)
        self.assertGreater(final_score, 0.60)

    def test_mi_numba_permutation(self):
        a = np.array([1, 0, 0, 0, 1, 1, 1, 0] * 10**3, dtype=np.int32)
        b = np.array(np.random.permutation(a), dtype=np.int32)
        final_score = mutual_info_estimator_numba(a, b, np.float32(1.0), False)
        self.assertLess(final_score, 0.05)

    def test_mi_numba_interaction(self):
        # Let's create incrementally more noisy features and compare
        a = np.array([1, 0, 0, 0, 1, 1, 1, 0], dtype=np.int32)
        lowest = np.array(np.random.permutation(a), dtype=np.int32)
        medium = np.array([1, 1, 0, 0, 1, 1, 1, 1], dtype=np.int32)
        high = np.array([1, 0, 0, 0, 1, 1, 1, 1], dtype=np.int32)

        lowest_score = mutual_info_estimator_numba(
            a, lowest, np.float32(1.0), False,
        )
        medium_score = mutual_info_estimator_numba(
            a, medium, np.float32(1.0), False,
        )
        high_score = mutual_info_estimator_numba(
            a, high, np.float32(1.0), False,
        )

        scores = [lowest_score, medium_score, high_score]
        sorted_score_indices = np.argsort(scores)
        self.assertEqual(np.sum(np.array([0, 1, 2]) - sorted_score_indices), 0)

    def test_mi_numba_higher_order(self):
        # The famous xor test
        vector_first = np.round(np.random.random(1000)).astype(np.int32)
        vector_second = np.round(np.random.random(1000)).astype(np.int32)
        vector_third = np.logical_xor(
            vector_first, vector_second,
        ).astype(np.int32)

        score_independent_first = mutual_info_estimator_numba(
            vector_first, vector_third, np.float32(1.0), False,
        )

        score_independent_second = mutual_info_estimator_numba(
            vector_second, vector_third, np.float32(1.0), False,
        )

        # This must be very close to zero/negative
        self.assertLess(score_independent_first, 0.01)
        self.assertLess(score_independent_second, 0.01)

        # --interaction_order 2 simulation
        combined_feature = np.array(
            list(hash(x) for x in zip(vector_first, vector_second)),
        ).astype(np.int32)

        score_combined = mutual_info_estimator_numba(
            combined_feature, vector_third, np.float32(1.0), False,
        )

        # This must be in the range of identity
        self.assertGreater(score_combined, 0.60)
    
    # === NEW COMPREHENSIVE TESTS ===
    
    def test_empty_arrays(self):
        """Test behavior with empty arrays"""
        a = np.array([], dtype=np.int32)
        b = np.array([], dtype=np.int32)
        
        # Should handle empty arrays gracefully
        with self.assertRaises((IndexError, ValueError)):
            mutual_info_estimator_numba(a, b, np.float32(1.0), False)
    
    def test_single_element_arrays(self):
        """Test arrays with single elements"""
        a = np.array([1], dtype=np.int32)
        b = np.array([0], dtype=np.int32)
        
        # Single element arrays should work
        result = mutual_info_estimator_numba(a, b, np.float32(1.0), False)
        self.assertIsInstance(result, (float, np.float32))
    
    def test_identical_arrays(self):
        """Test perfectly correlated arrays"""
        a = np.array([1, 2, 3, 1, 2, 3] * 100, dtype=np.int32)
        b = a.copy()
        
        result = mutual_info_estimator_numba(a, b, np.float32(1.0), False)
        # Identical arrays should have high mutual information
        self.assertGreater(result, 0.5)
    
    def test_approximation_factors(self):
        """Test different approximation factors"""
        a = np.array([1, 0, 1, 0, 1, 0] * 1000, dtype=np.int32)
        b = np.array([0, 1, 0, 1, 0, 1] * 1000, dtype=np.int32)
        
        # Test various approximation factors
        for factor in [0.1, 0.5, 1.0]:
            result = mutual_info_estimator_numba(a, b, np.float32(factor), False)
            self.assertIsInstance(result, (float, np.float32))
            
    def test_approximation_factor_edge_cases(self):
        """Test edge cases for approximation factor"""
        a = np.array([1, 0, 1, 0] * 100, dtype=np.int32)
        b = np.array([0, 1, 0, 1] * 100, dtype=np.int32)
        
        # Very small approximation factor
        result = mutual_info_estimator_numba(a, b, np.float32(0.01), False)
        self.assertIsInstance(result, (float, np.float32))
        
        # Approximation factor > 1 (should still work)
        result = mutual_info_estimator_numba(a, b, np.float32(1.5), False)
        self.assertIsInstance(result, (float, np.float32))
    
    def test_cardinality_correction(self):
        """Test cardinality correction flag"""
        a = np.array([1, 0, 1, 0, 1, 0] * 500, dtype=np.int32)
        b = np.array([1, 0, 1, 0, 1, 0] * 500, dtype=np.int32)
        
        # Without cardinality correction
        result_no_corr = mutual_info_estimator_numba(a, b, np.float32(1.0), False)
        
        # With cardinality correction  
        result_with_corr = mutual_info_estimator_numba(a, b, np.float32(1.0), True)
        
        # Both should be valid but may differ
        self.assertIsInstance(result_no_corr, (float, np.float32))
        self.assertIsInstance(result_with_corr, (float, np.float32))
    
    def test_different_array_lengths(self):
        """Test arrays of different lengths (should fail)"""
        a = np.array([1, 0, 1], dtype=np.int32)
        b = np.array([0, 1], dtype=np.int32)
        
        with self.assertRaises((IndexError, ValueError)):
            mutual_info_estimator_numba(a, b, np.float32(1.0), False)
    
    def test_binary_vs_multiclass(self):
        """Test binary vs multiclass scenarios"""
        # Binary case
        a_binary = np.array([0, 1] * 500, dtype=np.int32)
        b_binary = np.array([1, 0] * 500, dtype=np.int32)
        
        result_binary = mutual_info_estimator_numba(a_binary, b_binary, np.float32(1.0), False)
        
        # Multiclass case
        a_multi = np.array([0, 1, 2] * 333 + [0], dtype=np.int32)
        b_multi = np.array([2, 0, 1] * 333 + [1], dtype=np.int32)
        
        result_multi = mutual_info_estimator_numba(a_multi, b_multi, np.float32(1.0), False)
        
        # Both should be valid
        self.assertIsInstance(result_binary, (float, np.float32))
        self.assertIsInstance(result_multi, (float, np.float32))
    
    def test_extreme_values(self):
        """Test with extreme integer values"""
        max_val = np.iinfo(np.int32).max
        a = np.array([0, max_val] * 100, dtype=np.int32)
        b = np.array([max_val, 0] * 100, dtype=np.int32)
        
        result = mutual_info_estimator_numba(a, b, np.float32(1.0), False)
        self.assertIsInstance(result, (float, np.float32))
    
    def test_all_same_values(self):
        """Test arrays where all values are the same"""
        a = np.array([5] * 1000, dtype=np.int32)
        b = np.array([5] * 1000, dtype=np.int32)
        
        result = mutual_info_estimator_numba(a, b, np.float32(1.0), False)
        # Should handle constant arrays
        self.assertIsInstance(result, (float, np.float32))
    
    def test_large_arrays_performance(self):
        """Test with large arrays for performance validation"""
        size = 50000
        a = np.random.randint(0, 10, size=size, dtype=np.int32)
        b = np.random.randint(0, 10, size=size, dtype=np.int32)
        
        result = mutual_info_estimator_numba(a, b, np.float32(0.1), True)
        self.assertIsInstance(result, (float, np.float32))
    
    def test_deterministic_behavior(self):
        """Test that results are deterministic for same inputs"""
        a = np.array([1, 0, 1, 0, 1] * 200, dtype=np.int32)
        b = np.array([0, 1, 0, 1, 0] * 200, dtype=np.int32)
        
        # Multiple runs should give same result
        result1 = mutual_info_estimator_numba(a, b, np.float32(1.0), False)
        result2 = mutual_info_estimator_numba(a, b, np.float32(1.0), False) 
        result3 = mutual_info_estimator_numba(a, b, np.float32(1.0), False)
        
        self.assertEqual(result1, result2)
        self.assertEqual(result2, result3)
    
    def test_independence_detection(self):
        """Test detection of statistical independence"""
        np.random.seed(42)  # For reproducible randomness
        
        # Create independent variables
        a = np.random.randint(0, 3, size=5000, dtype=np.int32)
        b = np.random.randint(0, 3, size=5000, dtype=np.int32) 
        
        result = mutual_info_estimator_numba(a, b, np.float32(1.0), False)
        
        # Independent variables should have low mutual information
        # Note: Due to finite sample effects, may not be exactly 0
        self.assertLess(abs(result), 0.2)
    
    def test_functional_relationship(self):
        """Test detection of functional relationships"""
        # Y = f(X) relationship
        a = np.array([0, 1, 2] * 1000, dtype=np.int32)
        b = np.array([0, 2, 4] * 1000, dtype=np.int32)  # b = 2*a
        
        result = mutual_info_estimator_numba(a, b, np.float32(1.0), False)
        
        # Functional relationship should have high mutual information
        self.assertGreater(result, 0.5)
    
    def test_noise_robustness(self):
        """Test robustness to noise in relationship"""
        np.random.seed(999)
        
        # Base relationship
        a = np.array([0, 1] * 2500, dtype=np.int32)
        b_clean = a.copy()
        
        # Add noise (flip 10% of values)
        noise_indices = np.random.choice(len(b_clean), size=len(b_clean)//10, replace=False)
        b_noisy = b_clean.copy()
        b_noisy[noise_indices] = 1 - b_noisy[noise_indices]
        
        result_clean = mutual_info_estimator_numba(a, b_clean, np.float32(1.0), False)
        result_noisy = mutual_info_estimator_numba(a, b_noisy, np.float32(1.0), False)
        
        # Noisy version should have lower MI than clean version
        self.assertLess(result_noisy, result_clean)
        
        # But both should be positive
        self.assertGreater(result_clean, 0.4)
        self.assertGreater(result_noisy, 0.0)
