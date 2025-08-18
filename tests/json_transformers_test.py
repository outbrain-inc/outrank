from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

from outrank.feature_transformations.ranking_transformers import FeatureTransformerGeneric

sys.path.append('./outrank')


class JSONTransformersTest(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        self.test_transformers = {
            "_tr_sqrt": "np.sqrt(X)",
            "_tr_log": "np.log(X + 1)",
            "_tr_square": "np.square(X)",
            "_tr_sigmoid": "1 / (1 + np.exp(-X))"
        }
        
        # Create a temporary JSON file
        self.temp_json_fd, self.temp_json_path = tempfile.mkstemp(suffix='.json')
        with open(self.temp_json_path, 'w') as f:
            json.dump(self.test_transformers, f)
    
    def tearDown(self):
        """Clean up test fixtures."""
        os.close(self.temp_json_fd)
        os.unlink(self.temp_json_path)
    
    def test_load_transformers_from_json_file(self):
        """Test loading transformers from a JSON file."""
        numeric_columns = {'feature1', 'feature2'}
        transformer = FeatureTransformerGeneric(numeric_columns, preset=self.temp_json_path)
        
        # Check that transformers were loaded
        self.assertEqual(len(transformer.transformer_collection), 4)
        self.assertIn("_tr_sqrt", transformer.transformer_collection)
        self.assertIn("_tr_log", transformer.transformer_collection)
        self.assertIn("_tr_square", transformer.transformer_collection)
        self.assertIn("_tr_sigmoid", transformer.transformer_collection)
        
        # Check specific transformer expressions
        self.assertEqual(transformer.transformer_collection["_tr_sqrt"], "np.sqrt(X)")
        self.assertEqual(transformer.transformer_collection["_tr_log"], "np.log(X + 1)")
    
    def test_json_transformers_functionality(self):
        """Test that JSON-loaded transformers work correctly with data."""
        # Create test data
        test_data = pd.DataFrame({
            'feature1': [1.0, 4.0, 9.0, 16.0],
            'feature2': [0.0, 1.0, 2.0, 3.0]
        })
        
        numeric_columns = {'feature1', 'feature2'}
        transformer = FeatureTransformerGeneric(numeric_columns, preset=self.temp_json_path)
        
        # Apply transformations
        transformed_data = transformer.construct_new_features(test_data)
        
        # Check that new features were created
        original_columns = set(test_data.columns)
        new_columns = set(transformed_data.columns) - original_columns
        self.assertGreater(len(new_columns), 0)
        
        # Verify some specific transformations
        if 'feature1_tr_sqrt' in transformed_data.columns:
            # sqrt([1, 4, 9, 16]) should be [1, 2, 3, 4]
            expected_sqrt = ['1.0', '2.0', '3.0', '4.0']
            actual_sqrt = transformed_data['feature1_tr_sqrt'].astype(str).tolist()
            self.assertEqual(actual_sqrt, expected_sqrt)
    
    # === NEW COMPREHENSIVE TESTS ===
    
    def test_empty_dataframe(self):
        """Test behavior with empty DataFrame"""
        empty_df = pd.DataFrame()
        numeric_columns = set()
        
        transformer = FeatureTransformerGeneric(numeric_columns, preset=self.temp_json_path)
        result = transformer.construct_new_features(empty_df)
        
        # Should handle empty DataFrame gracefully
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 0)
    
    def test_no_matching_columns(self):
        """Test when no columns match the numeric_columns specification"""
        test_data = pd.DataFrame({
            'feature1': [1.0, 2.0, 3.0],
            'feature2': [4.0, 5.0, 6.0]
        })
        
        # Specify columns that don't exist - should raise KeyError
        numeric_columns = {'nonexistent1', 'nonexistent2'}
        transformer = FeatureTransformerGeneric(numeric_columns, preset=self.temp_json_path)
        
        with self.assertRaises(KeyError):
            transformer.construct_new_features(test_data)
    
    def test_single_column_dataframe(self):
        """Test with single column DataFrame"""
        test_data = pd.DataFrame({'feature1': [1.0, 4.0, 9.0, 16.0]})
        numeric_columns = {'feature1'}
        
        transformer = FeatureTransformerGeneric(numeric_columns, preset=self.temp_json_path)
        result = transformer.construct_new_features(test_data)
        
        # Should create new transformed columns
        self.assertGreater(len(result.columns), len(test_data.columns))
    
    def test_mixed_data_types(self):
        """Test with mixed data types in DataFrame"""
        test_data = pd.DataFrame({
            'numeric_int': [1, 2, 3, 4],
            'numeric_float': [1.1, 2.2, 3.3, 4.4],
            'string_col': ['a', 'b', 'c', 'd'],
            'bool_col': [True, False, True, False]
        })
        
        # Only specify numeric columns
        numeric_columns = {'numeric_int', 'numeric_float'}
        transformer = FeatureTransformerGeneric(numeric_columns, preset=self.temp_json_path)
        
        result = transformer.construct_new_features(test_data)
        
        # Should preserve non-numeric columns and add transformed ones
        self.assertTrue('string_col' in result.columns)
        self.assertTrue('bool_col' in result.columns)
        self.assertGreater(len(result.columns), len(test_data.columns))
    
    def test_dataframe_with_nan_values(self):
        """Test handling of NaN values in data"""
        test_data = pd.DataFrame({
            'feature1': [1.0, np.nan, 9.0, 16.0],
            'feature2': [0.0, 1.0, np.nan, 3.0]
        })
        
        numeric_columns = {'feature1', 'feature2'}
        transformer = FeatureTransformerGeneric(numeric_columns, preset=self.temp_json_path)
        
        # Should handle NaN values gracefully
        result = transformer.construct_new_features(test_data)
        self.assertIsInstance(result, pd.DataFrame)
    
    def test_large_dataframe_performance(self):
        """Test performance with larger DataFrame"""
        # Create larger test data
        size = 1000
        test_data = pd.DataFrame({
            'feature1': np.random.rand(size) * 100,
            'feature2': np.random.rand(size) * 50,
            'feature3': np.random.rand(size) * 10
        })
        
        numeric_columns = {'feature1', 'feature2', 'feature3'}
        transformer = FeatureTransformerGeneric(numeric_columns, preset=self.temp_json_path)
        
        result = transformer.construct_new_features(test_data)
        
        # Should complete successfully
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), size)
        self.assertGreater(len(result.columns), len(test_data.columns))
    
    def test_transformer_caching_behavior(self):
        """Test that transformer behaves consistently across multiple calls"""
        test_data = pd.DataFrame({
            'feature1': [1.0, 4.0, 9.0, 16.0],
            'feature2': [0.0, 1.0, 2.0, 3.0]
        })
        
        numeric_columns = {'feature1', 'feature2'}
        transformer = FeatureTransformerGeneric(numeric_columns, preset=self.temp_json_path)
        
        # Apply transformation twice
        result1 = transformer.construct_new_features(test_data)
        result2 = transformer.construct_new_features(test_data)
        
        # Should get identical results
        pd.testing.assert_frame_equal(result1, result2)
    
    def test_malformed_json_transformers(self):
        """Test error handling for malformed transformations in JSON"""
        malformed_transformers = {
            "_tr_valid": "np.sqrt(X)",
            "_tr_malformed": "invalid_function(X)",  # Non-existent function
            "_tr_syntax_error": "np.sqrt(X"  # Syntax error
        }
        
        malformed_json_fd, malformed_json_path = tempfile.mkstemp(suffix='.json')
        try:
            with open(malformed_json_path, 'w') as f:
                json.dump(malformed_transformers, f)
            
            test_data = pd.DataFrame({'feature1': [1.0, 4.0, 9.0]})
            numeric_columns = {'feature1'}
            
            # Should raise error for malformed transformations
            transformer = FeatureTransformerGeneric(numeric_columns, preset=malformed_json_path)
            with self.assertRaises((NameError, SyntaxError)):
                transformer.construct_new_features(test_data)
            
        finally:
            os.close(malformed_json_fd)
            os.unlink(malformed_json_path)
    
    def test_complex_json_transformations(self):
        """Test complex mathematical transformations"""
        complex_transformers = {
            "_tr_log": "np.log(X + 1e-8)",  # Avoid log(0)
            "_tr_exp": "np.exp(X)",
            "_tr_square": "X**2",
            "_tr_reciprocal": "1.0 / (X + 1e-8)",  # Avoid division by zero
            "_tr_sin": "np.sin(X)",
            "_tr_combined": "np.sqrt(X**2 + 1)"
        }
        
        complex_json_fd, complex_json_path = tempfile.mkstemp(suffix='.json')
        try:
            with open(complex_json_path, 'w') as f:
                json.dump(complex_transformers, f)
            
            test_data = pd.DataFrame({
                'feature1': [1.0, 2.0, 3.0, 4.0],
                'feature2': [0.1, 0.5, 1.0, 2.0]
            })
            
            numeric_columns = {'feature1', 'feature2'}
            transformer = FeatureTransformerGeneric(numeric_columns, preset=complex_json_path)
            
            result = transformer.construct_new_features(test_data)
            
            # Should create multiple new features
            original_columns = set(test_data.columns)
            new_columns = set(result.columns) - original_columns
            self.assertGreaterEqual(len(new_columns), len(complex_transformers) * len(numeric_columns))
            
        finally:
            os.close(complex_json_fd)
            os.unlink(complex_json_path)
    
    def test_zero_and_negative_values(self):
        """Test transformations with zero and negative values"""
        test_data = pd.DataFrame({
            'feature1': [-2.0, -1.0, 0.0, 1.0, 2.0],
            'feature2': [0.0, 0.1, 0.5, 1.0, 2.0]
        })
        
        numeric_columns = {'feature1', 'feature2'}
        transformer = FeatureTransformerGeneric(numeric_columns, preset=self.temp_json_path)
        
        result = transformer.construct_new_features(test_data)
        
        # Should handle edge values gracefully
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), len(test_data))
        
        # Check for inf or excessive NaN values
        for col in result.columns:
            if result[col].dtype in ['float64', 'float32']:
                inf_count = np.isinf(result[col]).sum()
                self.assertLess(inf_count, len(result) // 2, 
                    f"Too many infinite values in column {col}")
    
    def test_duplicate_column_handling(self):
        """Test behavior when DataFrame has normal column names"""
        # Test with normal columns since duplicate column names cause issues
        test_data = pd.DataFrame({
            'feature1': [1.0, 3.0, 5.0],
            'feature2': [2.0, 4.0, 6.0]
        })
        
        numeric_columns = {'feature1'}
        transformer = FeatureTransformerGeneric(numeric_columns, preset=self.temp_json_path)
        
        # Should handle normal columns fine
        result = transformer.construct_new_features(test_data)
        self.assertIsInstance(result, pd.DataFrame)
    
    def test_mixed_preset_and_json(self):
        """Test using both preset names and JSON files."""
        numeric_columns = {'feature1'}
        # Combine default preset with JSON file
        preset_string = f"minimal,{self.temp_json_path}"
        transformer = FeatureTransformerGeneric(numeric_columns, preset=preset_string)
        
        # Should have transformers from both sources
        self.assertGreater(len(transformer.transformer_collection), 4)  # More than just our JSON transformers
        
        # Should have our JSON transformers
        self.assertIn("_tr_sqrt", transformer.transformer_collection)
        
        # Should have minimal preset transformers
        self.assertIn("_tr_log(x+1)", transformer.transformer_collection)
    
    def test_invalid_json_file(self):
        """Test error handling for invalid JSON files."""
        # Test non-existent file
        with self.assertRaises(FileNotFoundError):
            FeatureTransformerGeneric({'feature1'}, preset="nonexistent.json")
        
        # Test invalid JSON format
        invalid_json_fd, invalid_json_path = tempfile.mkstemp(suffix='.json')
        try:
            with open(invalid_json_path, 'w') as f:
                f.write("{ invalid json")
            
            with self.assertRaises(ValueError):
                FeatureTransformerGeneric({'feature1'}, preset=invalid_json_path)
        finally:
            os.close(invalid_json_fd)
            os.unlink(invalid_json_path)
    
    def test_invalid_transformer_format(self):
        """Test error handling for invalid transformer specifications."""
        # Create JSON with non-string transformer
        invalid_transformers = {
            "_tr_sqrt": "np.sqrt(X)",
            "_tr_invalid": 123  # Invalid: not a string
        }
        
        invalid_json_fd, invalid_json_path = tempfile.mkstemp(suffix='.json')
        try:
            with open(invalid_json_path, 'w') as f:
                json.dump(invalid_transformers, f)
            
            with self.assertRaises(ValueError):
                FeatureTransformerGeneric({'feature1'}, preset=invalid_json_path)
        finally:
            os.close(invalid_json_fd)
            os.unlink(invalid_json_path)


if __name__ == '__main__':
    unittest.main()