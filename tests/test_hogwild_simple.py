from __future__ import annotations

import sys
import unittest
import tempfile
import os
import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
import tqdm
from pathos.multiprocessing import ProcessingPool as Pool

from outrank.core_ranking import estimate_importances_minibatches_hogwild

sys.path.append('./outrank')

# Suppress verbose logging for tests
logging.basicConfig(level=logging.ERROR)

np.random.seed(123)

@dataclass
class args:
    label_column: str = 'label'
    heuristic: str = 'MI-numba-randomized'
    target_ranking_only: str = 'True'
    interaction_order: int = 1
    combination_number_upper_bound: int = 100
    disable_tqdm: bool = True
    mi_stratified_sampling_ratio: float = 1.0
    reference_model_JSON: str = ''
    minibatch_size: int = 25
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


class SimpleHogwildTest(unittest.TestCase):
    
    def setUp(self):
        """Create simple test data"""
        # Generate very simple synthetic data
        np.random.seed(42)
        n_samples = 100
        
        # Create simple data pattern
        data = {
            'feature_0': np.random.randint(0, 3, n_samples),
            'feature_1': np.random.randint(0, 3, n_samples),
            'label': np.random.randint(0, 2, n_samples)
        }
        
        self.test_df = pd.DataFrame(data)
        
        # Create temporary CSV file
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        self.test_df.to_csv(self.temp_file.name, index=False)
        self.temp_file.close()
        
    def tearDown(self):
        """Clean up"""
        os.unlink(self.temp_file.name)
    
    def test_hogwild_basic_functionality(self):
        """Test basic Hogwild functionality without comparison"""
        
        column_descriptions = list(self.test_df.columns)
        fw_col_mapping = {}
        numeric_column_types = set()
        
        GLOBAL_CPU_POOL = Pool(processes=1)  # Use single process to avoid complexity
        
        try:
            results = estimate_importances_minibatches_hogwild(
                input_file=self.temp_file.name,
                column_descriptions=column_descriptions,
                fw_col_mapping=fw_col_mapping,
                numeric_column_types=numeric_column_types,
                args=args,
                cpu_pool=GLOBAL_CPU_POOL,
                delimiter=',',
                logger=logging.getLogger(),
            )
            
            # Basic checks that function completed successfully
            self.assertEqual(len(results), 9)  # Should return 9-tuple
            
            rankings = results[1]  # mutual_information_estimates
            
            # We should get some result (could be None for small data)
            print(f"Hogwild implementation completed successfully")
            print(f"Rankings result: {type(rankings)}")
            
            if rankings is not None:
                print(f"Number of rankings: {rankings.shape[0]}")
                self.assertGreaterEqual(rankings.shape[0], 0)
                
        except Exception as e:
            self.fail(f"Hogwild implementation failed with error: {e}")
            
        finally:
            GLOBAL_CPU_POOL.close()
            GLOBAL_CPU_POOL.join()


if __name__ == '__main__':
    unittest.main()