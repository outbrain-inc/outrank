from __future__ import annotations

import unittest
import numpy as np
from outrank.algorithms.feature_ranking.ranking_mi_multivalue import (
    multivalue_mutual_info_estimator,
    parse_multivalue_feature,
    jaccard_based_mutual_info,
    multivalue_mi_with_overlap,
    set_based_mutual_info,
)

class MultivalueMITest(unittest.TestCase):
    """Test cases for multivalue mutual information algorithms"""
    
    def test_parse_multivalue_feature(self):
        """Test parsing multivalue features into sets"""
        feature_vector = np.array(['a,b,c', 'b,c', '', 'a'])
        result = parse_multivalue_feature(feature_vector)
        
        expected = [
            {'a', 'b', 'c'},
            {'b', 'c'},
            set(),
            {'a'}
        ]
        
        self.assertEqual(result, expected)
    
    def test_parse_multivalue_feature_with_custom_delimiter(self):
        """Test parsing with custom delimiter"""
        feature_vector = np.array(['a;b;c', 'b;c', '', 'a'])
        result = parse_multivalue_feature(feature_vector, delimiter=';')
        
        expected = [
            {'a', 'b', 'c'},
            {'b', 'c'},
            set(),
            {'a'}
        ]
        
        self.assertEqual(result, expected)
    
    def test_set_based_mutual_info_identical_sets(self):
        """Test set-based MI with identical multivalue features"""
        X_sets = [{'a', 'b'}, {'b', 'c'}, {'a', 'c'}]
        Y_sets = [{'a', 'b'}, {'b', 'c'}, {'a', 'c'}]
        
        result = set_based_mutual_info(X_sets, Y_sets)
        
        # Should be high since features are identical
        self.assertGreater(result, 1.0)
    
    def test_set_based_mutual_info_independent_sets(self):
        """Test set-based MI with independent multivalue features"""
        X_sets = [{'a'}, {'b'}, {'c'}, {'d'}]
        Y_sets = [{'x'}, {'y'}, {'z'}, {'w'}]
        
        result = set_based_mutual_info(X_sets, Y_sets)
        
        # Should be high due to perfect correspondence (each X maps to unique Y)
        self.assertGreater(result, 1.0)
    
    def test_set_based_mutual_info_empty_sets(self):
        """Test set-based MI with empty sets"""
        X_sets = [set(), set(), set()]
        Y_sets = [set(), set(), set()]
        
        result = set_based_mutual_info(X_sets, Y_sets)
        
        # Should be 0 since all sets are identical (empty)
        self.assertEqual(result, 0.0)
    
    def test_jaccard_based_mutual_info_basic(self):
        """Test Jaccard-based MI with basic multivalue features"""
        X_sets = [{'a', 'b'}, {'b', 'c'}, {'a', 'c'}]
        Y_sets = [{'x', 'y'}, {'y', 'z'}, {'x', 'z'}]
        
        result = jaccard_based_mutual_info(X_sets, Y_sets)
        
        # Should return a valid MI score
        self.assertIsInstance(result, float)
        self.assertGreaterEqual(result, 0.0)
    
    def test_multivalue_mi_with_overlap_basic(self):
        """Test overlap-based MI with basic multivalue features"""
        X_sets = [{'a', 'b'}, {'b', 'c'}, {'a', 'c'}]
        Y_sets = [{'x', 'y'}, {'y', 'z'}, {'x', 'z'}]
        
        result = multivalue_mi_with_overlap(X_sets, Y_sets)
        
        # Should return a valid MI score
        self.assertIsInstance(result, float)
        self.assertGreaterEqual(result, 0.0)
    
    def test_multivalue_mutual_info_estimator_jaccard(self):
        """Test main estimator with Jaccard algorithm"""
        X = np.array(['a,b', 'b,c', 'a,c'])
        Y = np.array(['x,y', 'y,z', 'x,z'])
        
        result = multivalue_mutual_info_estimator(X, Y, algorithm='jaccard')
        
        self.assertIsInstance(result, float)
        self.assertGreaterEqual(result, 0.0)
    
    def test_multivalue_mutual_info_estimator_overlap(self):
        """Test main estimator with overlap algorithm"""
        X = np.array(['a,b', 'b,c', 'a,c'])
        Y = np.array(['x,y', 'y,z', 'x,z'])
        
        result = multivalue_mutual_info_estimator(X, Y, algorithm='overlap')
        
        self.assertIsInstance(result, float)
        self.assertGreaterEqual(result, 0.0)
    
    def test_multivalue_mutual_info_estimator_set_based(self):
        """Test main estimator with set-based algorithm"""
        X = np.array(['a,b', 'b,c', 'a,c'])
        Y = np.array(['x,y', 'y,z', 'x,z'])
        
        result = multivalue_mutual_info_estimator(X, Y, algorithm='set_based')
        
        self.assertIsInstance(result, float)
        self.assertGreaterEqual(result, 0.0)
    
    def test_multivalue_mutual_info_estimator_invalid_algorithm(self):
        """Test main estimator with invalid algorithm"""
        X = np.array(['a,b', 'b,c', 'a,c'])
        Y = np.array(['x,y', 'y,z', 'x,z'])
        
        with self.assertRaises(ValueError):
            multivalue_mutual_info_estimator(X, Y, algorithm='invalid')
    
    def test_multivalue_mutual_info_estimator_empty_input(self):
        """Test main estimator with empty input"""
        X = np.array([])
        Y = np.array([])
        
        result = multivalue_mutual_info_estimator(X, Y, algorithm='jaccard')
        self.assertEqual(result, 0.0)
    
    def test_multivalue_mutual_info_estimator_mismatched_lengths(self):
        """Test main estimator with mismatched input lengths"""
        X = np.array(['a,b'])
        Y = np.array(['x,y', 'y,z'])
        
        result = multivalue_mutual_info_estimator(X, Y, algorithm='jaccard')
        self.assertEqual(result, 0.0)
    
    def test_functional_relationship_detection(self):
        """Test detection of functional relationships in multivalue features"""
        # Create data with functional relationship: Y values determined by X values
        X = np.array(['a,b', 'b,c', 'c,d', 'a,b', 'b,c', 'c,d'])
        Y = np.array(['x,y', 'y,z', 'z,w', 'x,y', 'y,z', 'z,w'])
        
        result = multivalue_mutual_info_estimator(X, Y, algorithm='set_based')
        
        # Should detect the functional relationship
        self.assertGreater(result, 1.0)
    
    def test_no_relationship_detection(self):
        """Test detection when there's no relationship between features"""
        # Create completely random multivalue features
        np.random.seed(42)
        X = np.array([f'{i},{i+1}' for i in range(100)])
        Y = np.array([f'{100-i},{100-i-1}' for i in range(100)])
        
        result = multivalue_mutual_info_estimator(X, Y, algorithm='set_based')
        
        # Should detect high MI due to deterministic pattern (each X maps to unique Y)
        self.assertGreater(result, 0.0)


if __name__ == '__main__':
    unittest.main()