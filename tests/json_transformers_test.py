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