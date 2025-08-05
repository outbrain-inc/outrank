from __future__ import annotations

import sys
import unittest
import tempfile
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import tqdm
from pathos.multiprocessing import ProcessingPool as Pool

from outrank.core_ranking import estimate_importances_minibatches
from outrank.core_ranking import estimate_importances_minibatches_hogwild

sys.path.append('./outrank')

np.random.seed(123)

@dataclass
class args:
    label_column: str = 'label'
    heuristic: str = 'MI-numba-randomized'
    target_ranking_only: str = 'True'
    interaction_order: int = 1
    combination_number_upper_bound: int = 1024
    disable_tqdm: bool = True
    mi_stratified_sampling_ratio: float = 1.0
    reference_model_JSON: str = ''
    minibatch_size: int = 50
    subsampling: int = 1
    num_threads: int = 2
    enable_hogwild_parallelism: str = 'True'
    missing_value_symbols: str = 'None,nan,NaN,NA,,'
    transformers: str = 'none'
    explode_multivalue_features: str = 'False'
    subfeature_mapping: str = 'False'
    include_noise_baseline_features: str = 'False'
    task: str = 'ranking'
    max_unique_hist_constraint: int = 30000
    rare_value_count_upper_bound: int = 1
    feature_set_focus: str = None
    data_source: str = 'csv-raw'


class HogwildRankingTest(unittest.TestCase):
    
    def setUp(self):
        """Create test data"""
        # Generate synthetic data
        np.random.seed(42)
        n_samples = 200
        n_features = 4
        
        # Create synthetic data with some pattern
        data = np.random.randint(0, 5, (n_samples, n_features))
        # Make last column (label) somewhat dependent on first column
        data[:, -1] = (data[:, 0] + np.random.randint(0, 2, n_samples)) % 3
        
        # Create DataFrame
        columns = [f'feature_{i}' for i in range(n_features-1)] + ['label']
        self.test_df = pd.DataFrame(data, columns=columns)
        
        # Create temporary CSV file
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        self.test_df.to_csv(self.temp_file.name, index=False)
        self.temp_file.close()
        
    def tearDown(self):
        """Clean up"""
        os.unlink(self.temp_file.name)
    
    def test_hogwild_vs_standard_consistency(self):
        """Test that Hogwild implementation produces consistent results with standard implementation"""
        
        # Parameters for both implementations
        column_descriptions = list(self.test_df.columns)
        fw_col_mapping = {}
        numeric_column_types = set()
        
        GLOBAL_CPU_POOL = Pool(processes=args.num_threads)
        
        try:
            # Test standard implementation
            args.enable_hogwild_parallelism = 'False'
            
            results_standard = estimate_importances_minibatches(
                input_file=self.temp_file.name,
                column_descriptions=column_descriptions,
                fw_col_mapping=fw_col_mapping,
                numeric_column_types=numeric_column_types,
                args=args,
                cpu_pool=GLOBAL_CPU_POOL,
                delimiter=',',
                logger=None,
            )
            
            # Test Hogwild implementation
            args.enable_hogwild_parallelism = 'True'
            
            results_hogwild = estimate_importances_minibatches_hogwild(
                input_file=self.temp_file.name,
                column_descriptions=column_descriptions,
                fw_col_mapping=fw_col_mapping,
                numeric_column_types=numeric_column_types,
                args=args,
                cpu_pool=GLOBAL_CPU_POOL,
                delimiter=',',
                logger=None,
            )
            
            # Both should return the same structure
            self.assertEqual(len(results_standard), len(results_hogwild))
            
            # Check that we got ranking results
            standard_rankings = results_standard[1]  # mutual_information_estimates
            hogwild_rankings = results_hogwild[1]
            
            # Both should produce rankings
            self.assertIsNotNone(standard_rankings)
            self.assertIsNotNone(hogwild_rankings)
            
            # Should have same number of feature pairs
            if standard_rankings is not None and hogwild_rankings is not None:
                self.assertEqual(standard_rankings.shape[0], hogwild_rankings.shape[0])
                self.assertEqual(standard_rankings.shape[1], hogwild_rankings.shape[1])
                
                # Column names should match
                self.assertTrue(all(standard_rankings.columns == hogwild_rankings.columns))
                
                print(f"Standard implementation produced {standard_rankings.shape[0]} rankings")
                print(f"Hogwild implementation produced {hogwild_rankings.shape[0]} rankings")
                
        finally:
            GLOBAL_CPU_POOL.close()
            GLOBAL_CPU_POOL.join()
    
    def test_hogwild_batch_processing(self):
        """Test that Hogwild implementation can handle multiple batches"""
        
        column_descriptions = list(self.test_df.columns)
        fw_col_mapping = {}
        numeric_column_types = set()
        
        # Set small batch size to force multiple batches
        args.minibatch_size = 25  # Should create multiple batches from 200 samples
        
        GLOBAL_CPU_POOL = Pool(processes=args.num_threads)
        
        try:
            results = estimate_importances_minibatches_hogwild(
                input_file=self.temp_file.name,
                column_descriptions=column_descriptions,
                fw_col_mapping=fw_col_mapping,
                numeric_column_types=numeric_column_types,
                args=args,
                cpu_pool=GLOBAL_CPU_POOL,
                delimiter=',',
                logger=None,
            )
            
            # Should successfully process multiple batches
            rankings = results[1]  # mutual_information_estimates
            self.assertIsNotNone(rankings)
            
            if rankings is not None:
                # Should have rankings for our features vs label
                self.assertGreater(rankings.shape[0], 0)
                print(f"Hogwild multi-batch processing produced {rankings.shape[0]} rankings")
                
        finally:
            GLOBAL_CPU_POOL.close()
            GLOBAL_CPU_POOL.join()


if __name__ == '__main__':
    unittest.main()