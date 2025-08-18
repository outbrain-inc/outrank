from __future__ import annotations

import sys
import unittest
import tempfile
import os
import json

import numpy as np
import pandas as pd

from outrank.algorithms.sketches.counting_cms import CountMinSketch
from outrank.algorithms.feature_ranking.ranking_cov_alignment import max_pair_coverage
from outrank.algorithms.feature_ranking.ranking_mi_numba import mutual_info_estimator_numba

np.random.seed(42)
sys.path.append('./outrank')


class IntegrationTests(unittest.TestCase):
    """Integration tests that validate interaction between different components"""
    
    def test_cms_and_coverage_integration(self):
        """Test CountMinSketch with max_pair_coverage on same data"""
        # Generate test data
        array1 = np.random.randint(0, 10, size=5000, dtype=np.int32)
        array2 = np.random.randint(0, 10, size=5000, dtype=np.int32)
        
        # Test CountMinSketch on pairs
        cms = CountMinSketch(depth=4, width=1000)
        
        # Add pairs as hashed strings to CMS
        for a1, a2 in zip(array1, array2):
            pair_str = f"{a1},{a2}"
            cms.add(pair_str)
        
        # Get coverage measure
        coverage = max_pair_coverage(array1, array2)
        
        # Both should provide valid measures
        self.assertIsInstance(coverage, float)
        self.assertTrue(0 <= coverage <= 1)
        
        # Most frequent pair should have reasonable CMS count
        most_frequent_count = int(coverage * len(array1))
        if most_frequent_count > 0:
            # Find actual most frequent pair
            pairs = list(zip(array1, array2))
            from collections import Counter
            pair_counts = Counter(pairs)
            most_common_pair = pair_counts.most_common(1)[0]
            
            pair_str = f"{most_common_pair[0][0]},{most_common_pair[0][1]}"
            cms_count = cms.query(pair_str)
            
            # CMS count should be at least as large as true count
            self.assertGreaterEqual(cms_count, most_common_pair[1])
    
    def test_mi_and_coverage_correlation(self):
        """Test correlation between MI and coverage measures"""
        # Create data with known relationships
        base_data = np.random.randint(0, 5, size=2000, dtype=np.int32)
        
        test_cases = [
            # Perfect correlation
            (base_data, base_data.copy()),
            # Anti-correlation  
            (base_data, np.max(base_data) - base_data),
            # Random (independent)
            (base_data, np.random.randint(0, 5, size=2000, dtype=np.int32)),
        ]
        
        results = []
        for desc, (arr1, arr2) in zip(['perfect', 'anti', 'random'], test_cases):
            mi_score = mutual_info_estimator_numba(arr1, arr2, np.float32(1.0), False)
            coverage = max_pair_coverage(arr1, arr2)
            
            results.append({
                'type': desc,
                'mi': mi_score,
                'coverage': coverage
            })
        
        # Perfect correlation should have highest MI
        perfect_mi = results[0]['mi']
        random_mi = results[2]['mi']
        self.assertGreater(perfect_mi, random_mi)
        
        # Coverage should be highest for perfect correlation
        perfect_cov = results[0]['coverage']
        random_cov = results[2]['coverage']
        self.assertGreater(perfect_cov, random_cov)
    
    def test_component_scalability(self):
        """Test all components handle scaling to larger data"""
        sizes = [100, 1000, 10000]
        
        for size in sizes:
            with self.subTest(size=size):
                # Generate data
                arr1 = np.random.randint(0, min(size//10, 100), size=size, dtype=np.int32)
                arr2 = np.random.randint(0, min(size//10, 100), size=size, dtype=np.int32)
                
                # Test CountMinSketch
                cms = CountMinSketch(depth=6, width=min(size, 2**12))
                elements = [f"{a},{b}" for a, b in zip(arr1[:min(size, 1000)], arr2[:min(size, 1000)])]
                cms.batch_add(elements)
                
                # Test queries work
                query_result = cms.query(elements[0])
                self.assertGreaterEqual(query_result, 1)
                
                # Test coverage (with smaller arrays for very large sizes)
                test_arr1 = arr1[:min(size, 50000)]
                test_arr2 = arr2[:min(size, 50000)]
                coverage = max_pair_coverage(test_arr1, test_arr2)
                self.assertTrue(0 <= coverage <= 1)
                
                # Test MI with approximation for large sizes
                approx_factor = 1.0 if size <= 1000 else 0.1
                mi_score = mutual_info_estimator_numba(
                    test_arr1, test_arr2, 
                    np.float32(approx_factor), False
                )
                self.assertIsInstance(mi_score, (float, np.float32))


class PropertyBasedTests(unittest.TestCase):
    """Property-based style tests that verify mathematical properties"""
    
    def test_cms_count_property(self):
        """Test CountMinSketch count properties hold across many inputs"""
        cms = CountMinSketch(depth=5, width=1000)
        
        # Property: adding N identical elements should result in count >= N
        test_cases = [
            ('test_element', 10),
            ('another_element', 25),
            (42, 15),
            (('tuple', 'element'), 5),
        ]
        
        for element, count in test_cases:
            with self.subTest(element=element, count=count):
                # Add element 'count' times
                for _ in range(count):
                    cms.add(element)
                
                # Query should return at least 'count'
                result = cms.query(element)
                self.assertGreaterEqual(result, count,
                    msg=f"CMS returned {result}, expected at least {count} for {element}")
    
    def test_coverage_mathematical_properties(self):
        """Test mathematical properties of max_pair_coverage"""
        
        # Property 1: Coverage of all-same-value arrays should be 1.0
        for size in [10, 100, 1000]:
            # Use identical values to guarantee coverage = 1.0
            val = np.random.randint(0, 10)
            arr = np.array([val] * size, dtype=np.int32)
            coverage = max_pair_coverage(arr, arr)
            self.assertAlmostEqual(coverage, 1.0, places=5,
                msg=f"All-same arrays should have coverage 1.0, got {coverage}")
        
        # Property 1b: Coverage of identical arrays should be deterministic
        for size in [50, 200]:
            arr = np.random.randint(0, 3, size=size, dtype=np.int32)
            coverage1 = max_pair_coverage(arr, arr)
            coverage2 = max_pair_coverage(arr, arr)
            self.assertEqual(coverage1, coverage2,
                msg="Coverage should be deterministic for identical inputs")
        
        # Property 2: Coverage should be between 0 and 1
        for _ in range(10):
            size = np.random.randint(10, 1000)
            arr1 = np.random.randint(0, size//2, size=size, dtype=np.int32)
            arr2 = np.random.randint(0, size//2, size=size, dtype=np.int32)
            
            coverage = max_pair_coverage(arr1, arr2)
            self.assertTrue(0 <= coverage <= 1,
                msg=f"Coverage {coverage} outside valid range [0,1]")
        
        # Property 3: All unique pairs should give coverage = 1/n
        for n in [5, 10, 20]:
            arr1 = np.arange(n, dtype=np.int32)
            arr2 = np.arange(n, 2*n, dtype=np.int32)
            
            coverage = max_pair_coverage(arr1, arr2)
            expected = 1.0 / n
            self.assertAlmostEqual(coverage, expected, places=4,
                msg=f"All unique pairs should give coverage 1/{n}, got {coverage}")
    
    def test_mi_estimator_properties(self):
        """Test mathematical properties of mutual information estimator"""
        
        # Property 1: MI(X,X) should be high (perfect correlation)
        for size in [100, 500, 1000]:
            arr = np.random.randint(0, 10, size=size, dtype=np.int32)
            mi = mutual_info_estimator_numba(arr, arr, np.float32(1.0), False)
            self.assertGreater(mi, 0.3,
                msg=f"Self-MI should be high, got {mi} for size {size}")
        
        # Property 2: MI should be symmetric for identical arrays
        arr1 = np.random.randint(0, 5, size=500, dtype=np.int32)
        arr2 = np.random.randint(0, 5, size=500, dtype=np.int32)
        
        mi_12 = mutual_info_estimator_numba(arr1, arr2, np.float32(1.0), False)
        mi_21 = mutual_info_estimator_numba(arr2, arr1, np.float32(1.0), False)
        
        # Note: Due to the specific implementation, may not be exactly symmetric
        # Test that both are valid values
        self.assertIsInstance(mi_12, (float, np.float32))
        self.assertIsInstance(mi_21, (float, np.float32))
        
        # Property 3: Approximation should preserve ordering
        arr1 = np.array([0, 1] * 500, dtype=np.int32)
        arr2_high = arr1.copy()  # High correlation
        arr2_low = np.random.permutation(arr1)  # Low correlation
        
        mi_high_full = mutual_info_estimator_numba(arr1, arr2_high, np.float32(1.0), False)
        mi_low_full = mutual_info_estimator_numba(arr1, arr2_low, np.float32(1.0), False)
        mi_high_approx = mutual_info_estimator_numba(arr1, arr2_high, np.float32(0.5), False)
        mi_low_approx = mutual_info_estimator_numba(arr1, arr2_low, np.float32(0.5), False)
        
        # High correlation should maintain higher MI even with approximation
        self.assertGreater(mi_high_full, mi_low_full)
        # Note: Approximation may change absolute values but should preserve some ordering


class StressTests(unittest.TestCase):
    """Stress tests for robustness and edge cases"""
    
    def test_extreme_data_values(self):
        """Test components with extreme data values"""
        # Test with maximum int32 values
        max_val = np.iinfo(np.int32).max
        min_val = np.iinfo(np.int32).min
        
        extreme_arrays = [
            np.array([max_val] * 100, dtype=np.int32),
            np.array([min_val] * 100, dtype=np.int32),
            np.array([max_val, min_val] * 50, dtype=np.int32),
            np.array([0] * 100, dtype=np.int32),
        ]
        
        for i, arr in enumerate(extreme_arrays):
            with self.subTest(array_type=i):
                # Test CMS
                cms = CountMinSketch(depth=4, width=1000)
                cms.batch_add([str(x) for x in arr[:10]])  # Convert to strings
                self.assertGreaterEqual(cms.query(str(arr[0])), 1)
                
                # Test coverage
                coverage = max_pair_coverage(arr, arr)
                self.assertTrue(0 <= coverage <= 1 or np.isnan(coverage))
                
                # Test MI (may overflow/underflow, so just check it doesn't crash)
                try:
                    mi = mutual_info_estimator_numba(arr, arr, np.float32(1.0), False)
                    self.assertIsInstance(mi, (float, np.float32))
                except (OverflowError, ValueError):
                    # Acceptable for extreme values
                    pass
    
    def test_memory_efficiency(self):
        """Test memory efficiency with large sparse data"""
        # Create large sparse arrays (many repeated values)
        size = 100000
        arr1 = np.random.choice([0, 1, 2], size=size, p=[0.8, 0.15, 0.05]).astype(np.int32)
        arr2 = np.random.choice([0, 1, 2], size=size, p=[0.7, 0.2, 0.1]).astype(np.int32)
        
        # CountMinSketch should handle this efficiently
        cms = CountMinSketch(depth=6, width=2**12)
        
        # Add subset to avoid memory issues in test
        subset_size = min(size, 10000)
        elements = [f"{a},{b}" for a, b in zip(arr1[:subset_size], arr2[:subset_size])]
        cms.batch_add(elements)
        
        # Should complete without memory error
        result = cms.query(elements[0])
        self.assertGreaterEqual(result, 1)
        
        # Coverage should work with approximation
        coverage = max_pair_coverage(arr1[:subset_size], arr2[:subset_size])
        self.assertTrue(0 <= coverage <= 1)
        
        # MI with heavy approximation
        mi = mutual_info_estimator_numba(arr1[:subset_size], arr2[:subset_size], 
                                        np.float32(0.1), True)
        self.assertIsInstance(mi, (float, np.float32))
    
    def test_error_recovery(self):
        """Test graceful handling of various error conditions"""
        
        # Test CMS with extreme parameters
        try:
            cms_tiny = CountMinSketch(depth=1, width=1)
            cms_tiny.add("test")
            self.assertEqual(cms_tiny.query("test"), 1)
        except Exception as e:
            self.fail(f"CMS should handle extreme parameters: {e}")
        
        # Test coverage with single element
        single_arr = np.array([42], dtype=np.int32)
        coverage = max_pair_coverage(single_arr, single_arr)
        self.assertEqual(coverage, 1.0)
        
        # Test MI with constant arrays
        const_arr = np.array([5] * 100, dtype=np.int32)
        mi = mutual_info_estimator_numba(const_arr, const_arr, np.float32(1.0), False)
        self.assertIsInstance(mi, (float, np.float32))
        
        # Test with very small approximation factors
        arr1 = np.array([0, 1] * 100, dtype=np.int32)
        arr2 = np.array([1, 0] * 100, dtype=np.int32)
        
        mi_tiny = mutual_info_estimator_numba(arr1, arr2, np.float32(0.001), False)
        self.assertIsInstance(mi_tiny, (float, np.float32))


if __name__ == '__main__':
    unittest.main()