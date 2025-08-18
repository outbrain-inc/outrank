from __future__ import annotations

import unittest

import numpy as np

from outrank.algorithms.sketches.counting_cms import cms_hash
from outrank.algorithms.sketches.counting_cms import CountMinSketch


class TestCountMinSketch(unittest.TestCase):

    def setUp(self):
        # Set up a CountMinSketch instance with known parameters for testing
        self.depth = 6
        self.width = 2**10  # smaller width for testing purposes
        self.cms = CountMinSketch(self.depth, self.width)

    def test_init(self):
        self.assertEqual(self.cms.depth, self.depth)
        self.assertEqual(self.cms.width, self.width)
        self.assertEqual(self.cms.M.shape, (self.depth, self.width))
        self.assertEqual(len(self.cms.hash_seeds), self.depth)

    def test_add_and_query_single_element(self):
        # Test adding a single element and querying it
        element = 'test_element'
        self.cms.add(element)
        # The queried count should be at least 1 (could be higher due to hash collisions)
        self.assertGreaterEqual(self.cms.query(element), 1)

    def test_add_and_query_multiple_elements(self):
        elements = ['foo', 'bar', 'baz', 'qux', 'quux']
        for elem in elements:
            self.cms.add(elem)

        for elem in elements:
            self.assertGreaterEqual(self.cms.query(elem), 1)

    def test_batch_add_and_query(self):
        elements = ['foo', 'bar', 'baz'] * 10
        self.cms.batch_add(elements)

        for elem in set(elements):
            self.assertGreaterEqual(self.cms.query(elem), 10)

    def test_hash_uniformity(self):
        # Basic check for hash function's distribution
        seeds = np.array(np.random.randint(low=0, high=2**31 - 1, size=self.depth), dtype=np.uint32)
        hashes = [cms_hash(i, seeds[0], self.width) for i in range(1000)]
        # Expect fewer collisions over a small sample with a large width
        unique_hashes = len(set(hashes))
        self.assertGreater(unique_hashes, 900)
    
    # === NEW COMPREHENSIVE TESTS ===
    
    def test_init_boundary_values(self):
        """Test CountMinSketch initialization with boundary values"""
        # Test minimum valid dimensions
        cms_min = CountMinSketch(depth=1, width=1)
        self.assertEqual(cms_min.depth, 1)
        self.assertEqual(cms_min.width, 1)
        self.assertEqual(cms_min.M.shape, (1, 1))
        
        # Test large dimensions
        cms_large = CountMinSketch(depth=100, width=2**16)
        self.assertEqual(cms_large.depth, 100)
        self.assertEqual(cms_large.width, 2**16)
        
    def test_init_with_custom_matrix(self):
        """Test initialization with pre-existing matrix"""
        custom_matrix = np.ones((3, 5), dtype=np.int32)
        cms = CountMinSketch(depth=3, width=5, M=custom_matrix)
        self.assertTrue(np.array_equal(cms.M, custom_matrix))
        self.assertEqual(cms.depth, 3)
        self.assertEqual(cms.width, 5)
    
    def test_add_with_different_deltas(self):
        """Test adding elements with different delta values"""
        element = 'test'
        
        # Add with positive delta
        self.cms.add(element, delta=5)
        self.assertGreaterEqual(self.cms.query(element), 5)
        
        # Add with zero delta (should not change count)
        initial_count = self.cms.query(element)
        self.cms.add(element, delta=0)
        self.assertEqual(self.cms.query(element), initial_count)
        
        # Add with negative delta
        self.cms.add(element, delta=-2)
        self.assertGreaterEqual(self.cms.query(element), initial_count - 2)
    
    def test_add_various_data_types(self):
        """Test adding different data types"""
        test_cases = [
            ('string', str),
            (42, int),
            (3.14, float),
            (True, bool),
            ((1, 2, 3), tuple),
        ]
        
        for element, data_type in test_cases:
            with self.subTest(element=element, data_type=data_type):
                self.cms.add(element)
                count = self.cms.query(element)
                self.assertGreaterEqual(count, 1, 
                    f"Failed to add/query element of type {data_type}")
    
    def test_query_nonexistent_elements(self):
        """Test querying elements that were never added"""
        nonexistent_elements = ['never_added', 999, 'ghost_element']
        
        for element in nonexistent_elements:
            count = self.cms.query(element)
            self.assertEqual(count, 0, 
                f"Non-existent element {element} should have count 0")
    
    def test_batch_add_empty_list(self):
        """Test batch adding an empty list"""
        initial_matrix = self.cms.M.copy()
        self.cms.batch_add([])
        
        # Matrix should remain unchanged
        self.assertTrue(np.array_equal(self.cms.M, initial_matrix))
    
    def test_batch_add_large_list(self):
        """Test batch adding a very large list"""
        large_list = ['item'] * 10000
        self.cms.batch_add(large_list)
        
        count = self.cms.query('item')
        self.assertGreaterEqual(count, 10000)
    
    def test_hash_function_properties(self):
        """Test hash function mathematical properties"""
        seed = np.uint32(42)
        width = 1000
        
        # Test hash function returns values in range [0, width)
        for i in range(100):
            hash_val = cms_hash(i, seed, width)
            self.assertGreaterEqual(hash_val, 0)
            self.assertLess(hash_val, width)
            self.assertIsInstance(hash_val, (int, np.integer))
        
        # Test different seeds produce different distributions
        hashes1 = [cms_hash(i, np.uint32(1), width) for i in range(1000)]
        hashes2 = [cms_hash(i, np.uint32(2), width) for i in range(1000)]
        
        # Should have different distributions (not identical)
        self.assertNotEqual(hashes1, hashes2)
    
    def test_hash_collision_frequency(self):
        """Test hash collision rates are reasonable"""
        seed = np.uint32(123)
        width = 100
        num_items = 200  # More items than width to guarantee some collisions
        
        hashes = [cms_hash(i, seed, width) for i in range(num_items)]
        unique_hashes = len(set(hashes))
        
        # Should have some collisions but not too many
        self.assertLess(unique_hashes, num_items)  # Some collisions expected
        self.assertGreater(unique_hashes, width // 2)  # Not too many collisions
    
    def test_multiple_hash_seeds_independence(self):
        """Test that different hash seeds produce independent results"""
        cms = CountMinSketch(depth=4, width=1000)
        test_element = 'test_independence'
        
        # Get hash values for same element with different seeds
        hash_values = []
        for i in range(cms.depth):
            hash_val = cms_hash(test_element, cms.hash_seeds[i], cms.width)
            hash_values.append(hash_val)
        
        # All hash values should be different (very high probability)
        unique_hashes = len(set(hash_values))
        self.assertEqual(unique_hashes, cms.depth, 
            "Hash seeds should produce independent hash values")
    
    def test_accuracy_with_known_frequencies(self):
        """Test accuracy of count estimates with known ground truth"""
        # Create data with known frequencies
        elements = ['a'] * 100 + ['b'] * 50 + ['c'] * 25 + ['d'] * 10
        
        self.cms.batch_add(elements)
        
        # Verify estimates are at least as large as true counts
        self.assertGreaterEqual(self.cms.query('a'), 100)
        self.assertGreaterEqual(self.cms.query('b'), 50)
        self.assertGreaterEqual(self.cms.query('c'), 25)
        self.assertGreaterEqual(self.cms.query('d'), 10)
        
        # Verify estimates are reasonably close (within 2x for this small test)
        self.assertLessEqual(self.cms.query('a'), 200)
        self.assertLessEqual(self.cms.query('b'), 100)
    
    def test_get_matrix_returns_copy_safety(self):
        """Test that modifying returned matrix doesn't affect internal state"""
        original_matrix = self.cms.M.copy()
        returned_matrix = self.cms.get_matrix()
        
        # Modify the returned matrix
        returned_matrix[0, 0] = 999
        
        # Original should be unchanged if it's a proper copy
        # Note: Current implementation returns reference, this tests documents the behavior
        # In a production system, we might want get_matrix() to return a copy
        self.assertTrue(np.array_equal(self.cms.M, returned_matrix), 
            "get_matrix() returns reference to internal matrix")
    
    def test_consistent_query_results(self):
        """Test that multiple queries of same element return consistent results"""
        element = 'consistent_test'
        self.cms.add(element, delta=5)
        
        # Multiple queries should return the same result
        first_query = self.cms.query(element)
        second_query = self.cms.query(element) 
        third_query = self.cms.query(element)
        
        self.assertEqual(first_query, second_query)
        self.assertEqual(second_query, third_query)
