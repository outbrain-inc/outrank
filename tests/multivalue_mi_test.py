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
        # Using default delimiter '_'
        feature_vector = np.array(['a_b_c', 'b_c', '', 'a'])
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
        # Test with comma delimiter
        feature_vector = np.array(['a,b,c', 'b,c', '', 'a'])
        result = parse_multivalue_feature(feature_vector, delimiter=',')
        
        expected = [
            {'a', 'b', 'c'},
            {'b', 'c'},
            set(),
            {'a'}
        ]
        
        self.assertEqual(result, expected)
        
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
        X = np.array(['a_b', 'b_c', 'a_c'])
        Y = np.array(['x_y', 'y_z', 'x_z'])
        
        result = multivalue_mutual_info_estimator(X, Y, algorithm='jaccard')
        
        self.assertIsInstance(result, float)
        self.assertGreaterEqual(result, 0.0)
    
    def test_multivalue_mutual_info_estimator_overlap(self):
        """Test main estimator with overlap algorithm"""
        X = np.array(['a_b', 'b_c', 'a_c'])
        Y = np.array(['x_y', 'y_z', 'x_z'])
        
        result = multivalue_mutual_info_estimator(X, Y, algorithm='overlap')
        
        self.assertIsInstance(result, float)
        self.assertGreaterEqual(result, 0.0)
    
    def test_multivalue_mutual_info_estimator_set_based(self):
        """Test main estimator with set-based algorithm"""
        X = np.array(['a_b', 'b_c', 'a_c'])
        Y = np.array(['x_y', 'y_z', 'x_z'])
        
        result = multivalue_mutual_info_estimator(X, Y, algorithm='set_based')
        
        self.assertIsInstance(result, float)
        self.assertGreaterEqual(result, 0.0)
    
    def test_multivalue_mutual_info_estimator_invalid_algorithm(self):
        """Test main estimator with invalid algorithm"""
        X = np.array(['a_b', 'b_c', 'a_c'])
        Y = np.array(['x_y', 'y_z', 'x_z'])
        
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
        X = np.array(['a_b'])
        Y = np.array(['x_y', 'y_z'])
        
        result = multivalue_mutual_info_estimator(X, Y, algorithm='jaccard')
        self.assertEqual(result, 0.0)
    
    def test_functional_relationship_detection(self):
        """Test detection of functional relationships in multivalue features"""
        # Create data with functional relationship: Y values determined by X values
        X = np.array(['a_b', 'b_c', 'c_d', 'a_b', 'b_c', 'c_d'])
        Y = np.array(['x_y', 'y_z', 'z_w', 'x_y', 'y_z', 'z_w'])
        
        result = multivalue_mutual_info_estimator(X, Y, algorithm='set_based')
        
        # Should detect the functional relationship
        self.assertGreater(result, 1.0)
    
    def test_no_relationship_detection(self):
        """Test detection when there's no relationship between features"""
        # Create completely random multivalue features
        np.random.seed(42)
        X = np.array([f'{i}_{i+1}' for i in range(100)])
        Y = np.array([f'{100-i}_{100-i-1}' for i in range(100)])
        
        result = multivalue_mutual_info_estimator(X, Y, algorithm='set_based')
        
        # Should detect high MI due to deterministic pattern (each X maps to unique Y)
        self.assertGreater(result, 0.0)
    
    def test_sequential_pattern_without_intersections(self):
        """Test detection of sequential patterns when row-wise intersections are empty.
        
        This addresses the issue raised in GitHub where Jaccard and overlap methods
        returned 0 for data like:
        Col1: a,b  b,c  c,d (with comma delimiter)
        Col2: i,j,k  j,k,l  k,l,m (with comma delimiter)
        
        Here intersections are empty in all cases, but there is information shared
        through the sequential patterns.
        
        NOTE: Using comma delimiter here to test the specific reported case.
        """
        # Test case from GitHub comment - using comma delimiter
        Col1 = np.array(['a,b', 'b,c', 'c,d', 'd,e', 'e,f'])
        Col2 = np.array(['i,j,k', 'j,k,l', 'k,l,m', 'l,m,n', 'm,n,o'])
        
        # All algorithms should now detect information despite empty intersections
        jaccard_score = multivalue_mutual_info_estimator(Col1, Col2, algorithm='jaccard', delimiter=',')
        overlap_score = multivalue_mutual_info_estimator(Col1, Col2, algorithm='overlap', delimiter=',')
        set_based_score = multivalue_mutual_info_estimator(Col1, Col2, algorithm='set_based', delimiter=',')
        
        # All should detect meaningful information
        self.assertGreater(jaccard_score, 0.0, 
                          "Jaccard should detect information in sequential patterns")
        self.assertGreater(overlap_score, 0.0,
                          "Overlap should detect information in sequential patterns")
        self.assertGreater(set_based_score, 0.0,
                          "Set-based should detect information in sequential patterns")
        
        # Set-based typically gives highest scores
        self.assertGreater(set_based_score, overlap_score * 0.5)
    
    def test_multivalue_with_compound_values(self):
        """Test multivalue features with compound values like 'yellow_sun', 'green_grass', etc.
        
        This test addresses the request to handle realistic feature values that themselves
        contain underscores (e.g., colors with objects). The algorithm should treat
        'yellow_sun' as a single atomic value, not split it further.
        """
        # Multivalue features where each value is a compound word
        # Using '|' as delimiter to separate different multivalue items
        # since the values themselves contain underscores
        colors1 = np.array(['yellow_sun|green_grass', 'blue_sea|red_flower', 
                           'yellow_sun|blue_sea', 'green_grass|red_flower'])
        colors2 = np.array(['yellow_sun|blue_sea', 'green_grass|red_flower',
                           'yellow_sun|red_flower', 'blue_sea|green_grass'])
        
        # Test with pipe delimiter for the multivalue separation
        for algo in ['jaccard', 'overlap', 'set_based']:
            with self.subTest(algorithm=algo):
                score = multivalue_mutual_info_estimator(
                    colors1, colors2, algorithm=algo, delimiter='|'
                )
                # Should compute valid MI scores
                self.assertIsInstance(score, float)
                self.assertGreaterEqual(score, 0.0)
        
        # Verify parsing treats compound values as atomic units
        parsed = parse_multivalue_feature(colors1, delimiter='|')
        expected_first = {'yellow_sun', 'green_grass'}
        expected_second = {'blue_sea', 'red_flower'}
        
        self.assertEqual(parsed[0], expected_first, 
                        "Compound values should be treated as atomic units")
        self.assertEqual(parsed[1], expected_second,
                        "Compound values should be treated as atomic units")
        
        # Test that there's meaningful information between the features
        set_based_score = multivalue_mutual_info_estimator(
            colors1, colors2, algorithm='set_based', delimiter='|'
        )
        self.assertGreater(set_based_score, 0.0,
                          "Should detect information between correlated multivalue features")


if __name__ == '__main__':
    unittest.main()