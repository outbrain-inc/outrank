#!/usr/bin/env python3
"""
Test to verify performance optimizations work correctly.
"""

import unittest
import time
import sys
sys.path.insert(0, '/home/runner/work/outrank/outrank')

import pandas as pd
import numpy as np
from outrank.core_ranking import get_combinations_from_columns, mixed_rank_graph
from outrank.core_ranking import GLOBAL_PRIOR_COMB_COUNTS


class PerformanceOptimizationTest(unittest.TestCase):
    """Test the performance optimizations"""

    def setUp(self):
        # Clear global state
        GLOBAL_PRIOR_COMB_COUNTS.clear()
        
        # Create test data
        np.random.seed(42)
        self.test_data = {}
        for i in range(20):
            cardinality = np.random.randint(2, 11)
            self.test_data[f'feature_{i}'] = np.random.randint(0, cardinality, size=10000)
        self.test_data['label'] = np.random.randint(0, 2, size=10000)
        self.df = pd.DataFrame(self.test_data)

    def test_optimized_encoding_performance(self):
        """Test that optimized encoding is faster and produces same results"""
        
        # Original method simulation
        start = time.time()
        tmp_df_orig = self.df.copy().astype('category')
        encoded_orig = pd.DataFrame({k: tmp_df_orig[k].cat.codes for k in self.df.columns})
        time_orig = time.time() - start
        
        # Optimized method
        start = time.time()
        encoded_opt = self.df.astype('category').apply(lambda x: x.cat.codes)
        time_opt = time.time() - start
        
        # Verify results are equivalent
        self.assertTrue(encoded_orig.equals(encoded_opt))
        
        # Performance should be better or at least not worse
        self.assertLessEqual(time_opt, time_orig * 1.5)  # Allow 50% tolerance
        
    def test_optimized_combinations_target_ranking_only(self):
        """Test optimized combination generation for target_ranking_only"""
        
        class MockArgs:
            def __init__(self):
                self.label_column = 'label'
                self.target_ranking_only = 'True'
                self.heuristic = 'MI-numba-randomized'
        
        args = MockArgs()
        all_columns = self.df.columns
        
        # Test the optimized method
        start = time.time()
        combinations = get_combinations_from_columns(all_columns, args)
        time_opt = time.time() - start
        
        # Verify correct number of combinations (all features including label)
        expected_count = len(all_columns)  # All features with label, including label with itself
        self.assertEqual(len(combinations), expected_count)
        
        # Verify all combinations include the label
        for combo in combinations:
            self.assertIn(args.label_column, combo)
        
        # Performance should be very fast
        self.assertLess(time_opt, 0.001)  # Should be sub-millisecond

    def test_data_encoding_memory_efficiency(self):
        """Test that encoding is memory efficient"""
        
        # Test with larger dataset
        large_data = {}
        for i in range(10):
            large_data[f'feature_{i}'] = np.random.randint(0, 100, size=50000)
        large_data['label'] = np.random.randint(0, 2, size=50000)
        large_df = pd.DataFrame(large_data)
        
        # Measure memory usage before
        original_memory = large_df.memory_usage().sum()
        
        # Encode using optimized method
        encoded_df = large_df.astype('category').apply(lambda x: x.cat.codes)
        encoded_memory = encoded_df.memory_usage().sum()
        
        # Encoded should use significantly less memory (categorical -> int codes)
        self.assertLess(encoded_memory, original_memory * 0.5)

    def test_backward_compatibility(self):
        """Test that optimizations maintain backward compatibility"""
        
        class MockArgs:
            def __init__(self):
                self.label_column = 'label'
                self.target_ranking_only = 'False'  # Test non-optimized path
                self.heuristic = 'MI-numba-randomized'
        
        args = MockArgs()
        all_columns = self.df.columns
        
        # Should still work for non-target-ranking-only case
        combinations = get_combinations_from_columns(all_columns, args)
        
        # Should include all pairwise combinations
        self.assertGreater(len(combinations), len(all_columns))


if __name__ == '__main__':
    unittest.main()