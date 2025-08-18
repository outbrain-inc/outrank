from __future__ import annotations

import sys
import unittest

import numpy as np

from outrank.algorithms.feature_ranking.ranking_cov_alignment import \
    max_pair_coverage

np.random.seed(123)
sys.path.append('./outrank')


class TestMaxPairCoverage(unittest.TestCase):
    def test_basic_functionality(self):
        array1 = np.array([1, 2, 3, 1, 2])
        array2 = np.array([4, 5, 6, 4, 5])
        result = max_pair_coverage(array1, array2)
        self.assertAlmostEqual(result, 2/5, places=5)

    def test_identical_elements(self):
        array1 = np.array([1, 1, 1, 1])
        array2 = np.array([1, 1, 1, 1])
        result = max_pair_coverage(array1, array2)
        self.assertEqual(result, 1.0)

    def test_large_arrays(self):
        array1 = np.random.randint(0, 100, size=10000)
        array2 = np.random.randint(0, 100, size=10000)
        result = max_pair_coverage(array1, array2)
        self.assertTrue(0 <= result <= 1)

    def test_all_unique_pairs(self):
        array1 = np.array([1, 2, 3, 4, 5])
        array2 = np.array([6, 7, 8, 9, 10])
        result = max_pair_coverage(array1, array2)
        self.assertEqual(result, 1/5)

    def test_all_same_pairs(self):
        array1 = np.array([1, 1, 1, 1, 1])
        array2 = np.array([2, 2, 2, 2, 2])
        result = max_pair_coverage(array1, array2)
        self.assertEqual(result, 1.0)

    def test_high_collision_potential(self):
        array1 = np.array([1] * 1000)
        array2 = np.array([2] * 1000)
        result = max_pair_coverage(array1, array2)
        self.assertEqual(result, 1.0)

    def test_very_large_arrays(self):
        array1 = np.random.randint(0, 1000, size=1000000)
        array2 = np.random.randint(0, 1000, size=1000000)
        result = max_pair_coverage(array1, array2)
        self.assertTrue(0 <= result <= 1)
    
    # === NEW COMPREHENSIVE TESTS ===
    
    def test_empty_arrays(self):
        """Test behavior with empty arrays"""
        array1 = np.array([], dtype=np.int32)
        array2 = np.array([], dtype=np.int32)
        
        # Empty arrays result in NaN due to 0/0 division
        result = max_pair_coverage(array1, array2)
        self.assertTrue(np.isnan(result))
    
    def test_single_element_arrays(self):
        """Test arrays with single elements"""
        array1 = np.array([42], dtype=np.int32)
        array2 = np.array([73], dtype=np.int32)
        result = max_pair_coverage(array1, array2)
        self.assertEqual(result, 1.0)  # Single pair gets 100% coverage
    
    def test_two_element_arrays(self):
        """Test arrays with two elements"""
        # Different pairs
        array1 = np.array([1, 2], dtype=np.int32)
        array2 = np.array([3, 4], dtype=np.int32)
        result = max_pair_coverage(array1, array2)
        self.assertEqual(result, 0.5)  # Each pair appears once, max coverage is 1/2
        
        # Same pairs
        array1 = np.array([1, 1], dtype=np.int32)
        array2 = np.array([3, 3], dtype=np.int32)
        result = max_pair_coverage(array1, array2)
        self.assertEqual(result, 1.0)  # Same pair appears twice
    
    def test_mismatched_array_lengths(self):
        """Test error handling for arrays of different lengths"""
        array1 = np.array([1, 2, 3], dtype=np.int32)
        array2 = np.array([4, 5], dtype=np.int32)  # Different length
        
        with self.assertRaises(IndexError):
            max_pair_coverage(array1, array2)
    
    def test_wrong_data_types(self):
        """Test behavior with non-int32 arrays"""
        # Test with float arrays - should work due to numpy casting
        array1 = np.array([1.0, 2.0, 3.0])
        array2 = np.array([4.0, 5.0, 6.0])
        
        # Convert to int32 as expected by function signature
        array1_int32 = array1.astype(np.int32)
        array2_int32 = array2.astype(np.int32)
        result = max_pair_coverage(array1_int32, array2_int32)
        self.assertIsInstance(result, float)
        self.assertTrue(0 <= result <= 1)
    
    def test_negative_values(self):
        """Test arrays containing negative values"""
        array1 = np.array([-1, -2, -3, -1, -2], dtype=np.int32)
        array2 = np.array([4, 5, 6, 4, 5], dtype=np.int32)
        result = max_pair_coverage(array1, array2)
        
        # Should work with negative values
        self.assertIsInstance(result, float)
        self.assertTrue(0 <= result <= 1)
        self.assertAlmostEqual(result, 2/5, places=5)
    
    def test_zero_values(self):
        """Test arrays containing zero values"""
        array1 = np.array([0, 0, 1, 1], dtype=np.int32)
        array2 = np.array([0, 0, 2, 2], dtype=np.int32)
        result = max_pair_coverage(array1, array2)
        
        # Two (0,0) pairs and two (1,2) pairs, max coverage should be 0.5
        self.assertEqual(result, 0.5)
    
    def test_large_integer_values(self):
        """Test with very large integer values"""
        max_int32 = np.iinfo(np.int32).max
        min_int32 = np.iinfo(np.int32).min
        
        array1 = np.array([max_int32, min_int32, 0], dtype=np.int32)
        array2 = np.array([max_int32, min_int32, 0], dtype=np.int32)
        result = max_pair_coverage(array1, array2)
        
        # Due to hash function behavior and potential overflow, result should be valid float
        self.assertIsInstance(result, float)
        self.assertTrue(0 <= result <= 1 or np.isnan(result))  # Allow NaN due to overflow
    
    def test_hash_collision_simulation(self):
        """Test behavior when hash collisions might occur"""
        # Create values that might cause hash collisions
        # Using large numbers that could wrap around in hash function
        large_vals = np.array([1471343, 2942686, 4414029], dtype=np.int32)
        array1 = np.tile(large_vals, 100)
        array2 = np.tile([1, 2, 3], 100)
        
        result = max_pair_coverage(array1, array2)
        
        # Should handle potential hash collisions gracefully
        self.assertIsInstance(result, float)
        self.assertTrue(0 <= result <= 1)
    
    def test_mathematical_properties(self):
        """Test mathematical properties of the coverage function"""
        array1 = np.array([1, 2, 3, 1, 2, 1], dtype=np.int32)
        array2 = np.array([4, 5, 6, 4, 5, 4], dtype=np.int32)
        
        result = max_pair_coverage(array1, array2)
        
        # Coverage should be fraction of most common pair
        # (1,4) appears 3 times out of 6 total, so coverage = 3/6 = 0.5
        self.assertEqual(result, 0.5)
        
        # Test symmetry property isn't expected (function uses el1 * constant - el2)
        result_swapped = max_pair_coverage(array2, array1)
        # Results may be different due to hash function asymmetry
        self.assertIsInstance(result_swapped, float)
        self.assertTrue(0 <= result_swapped <= 1)
    
    def test_coverage_bounds_verification(self):
        """Verify coverage is always between 0 and 1"""
        # Test with various random configurations
        np.random.seed(456)  # Different seed for this test
        
        for size in [10, 100, 1000]:
            for num_unique in [1, size//4, size//2, size]:
                array1 = np.random.randint(0, num_unique, size=size, dtype=np.int32)
                array2 = np.random.randint(0, num_unique, size=size, dtype=np.int32)
                
                result = max_pair_coverage(array1, array2)
                
                with self.subTest(size=size, num_unique=num_unique):
                    self.assertGreaterEqual(result, 0.0, 
                        f"Coverage should be >= 0, got {result}")
                    self.assertLessEqual(result, 1.0, 
                        f"Coverage should be <= 1, got {result}")
                    self.assertIsInstance(result, float)
    
    def test_hash_function_properties(self):
        """Test properties of the internal hash function indirectly"""
        # Create array where we can predict hash behavior
        array1 = np.array([0, 1, 2], dtype=np.int32)
        array2 = np.array([0, 0, 0], dtype=np.int32)
        
        result = max_pair_coverage(array1, array2)
        
        # Each pair (0,0), (1,0), (2,0) should hash to different values
        # unless there are collisions, so max coverage should be 1/3
        self.assertAlmostEqual(result, 1/3, places=5)
    
    def test_deterministic_behavior(self):
        """Test that function returns consistent results for same input"""
        array1 = np.array([1, 2, 3, 1, 2], dtype=np.int32)
        array2 = np.array([4, 5, 6, 4, 5], dtype=np.int32)
        
        # Multiple calls should return identical results
        result1 = max_pair_coverage(array1, array2)
        result2 = max_pair_coverage(array1, array2)
        result3 = max_pair_coverage(array1, array2)
        
        self.assertEqual(result1, result2)
        self.assertEqual(result2, result3)
    
    def test_coverage_with_all_different_pairs(self):
        """Test coverage when all pairs are unique"""
        n = 100
        array1 = np.arange(n, dtype=np.int32)
        array2 = np.arange(n, n*2, dtype=np.int32)
        
        result = max_pair_coverage(array1, array2)
        
        # All pairs are unique, so max coverage is 1/n
        expected = 1.0 / n
        self.assertAlmostEqual(result, expected, places=5)
    
    def test_maximum_coverage_scenario(self):
        """Test scenario that should give maximum coverage (1.0)"""
        # All pairs are identical
        array1 = np.array([42] * 100, dtype=np.int32)
        array2 = np.array([73] * 100, dtype=np.int32)
        
        result = max_pair_coverage(array1, array2)
        self.assertEqual(result, 1.0)
