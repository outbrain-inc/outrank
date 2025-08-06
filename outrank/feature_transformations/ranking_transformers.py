# A collection of feature transformers that can be considered
from __future__ import annotations

import json
import logging
import os
from typing import Any

import numpy as np
import pandas as pd

import outrank.feature_transformations.feature_transformer_vault as transformer_vault
from outrank.core_utils import internal_hash

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)


class FeatureTransformerNoise:
    def __init__(self):
        self.noise_preset = 'default'

    def construct_new_features(self, dataframe: pd.DataFrame, label_column=None):
        """Generate a few standard noise distributions"""

        new_columns = dict()
        if self.noise_preset == 'default':
            new_columns['CONTROL-constant0'] = np.array([0] * dataframe.shape[0])
            new_columns['CONTROL-gaussian'] = np.random.normal(
                size=dataframe.shape[0],
            )
            new_columns['CONTROL-uniform'] = np.random.random(
                dataframe.shape[0],
            )
            new_columns['CONTROL-random-binary'] = np.random.randint(
                0, 2, dataframe.shape[0],
            )
            new_columns['CONTROL-random-card100'] = np.random.randint(
                0, 1 + 1 * 10**2, dataframe.shape[0],
            )
            new_columns['CONTROL-random-card2k'] = np.random.randint(
                0, 1 + 2 * 10**3, dataframe.shape[0],
            )
            new_columns['CONTROL-random-card10k'] = np.random.randint(
                0, 1 + 10 * 10**3, dataframe.shape[0],
            )
            new_columns['CONTROL-random-card50k'] = np.random.randint(
                0, 1 + 50 * 10**3, dataframe.shape[0],
            )
            new_columns['CONTROL-int-sequence'] = np.arange(
                0, dataframe.shape[0], 1.0,
            )

            if label_column not in dataframe.columns:
                logging.warn(
                    'Could not find target feature in your data set - please inspect the columns if doing targeted ranking!',
                )
            else:
                new_columns['CONTROL-target'] = dataframe[label_column]

            new_columns['CONTROL-volume'] = np.array([
                internal_hash(str(x)) for _, x in dataframe.iterrows()
            ])
        else:
            # Not relevant yet; will be if this is useful.
            pass

        if len(new_columns) > 0:
            tmp_df = pd.DataFrame(new_columns)
            dataframe = pd.concat([dataframe, tmp_df], axis=1)
            del tmp_df

        return dataframe


class FeatureTransformerGeneric:
    def __init__(self, numeric_column_names: set[str], preset: str = 'default'):
        self.transformer_collection: dict[str, str] = dict()
        
        for transformer_namespace in preset.split(','):
            transformer_namespace = transformer_namespace.strip()
            
            # Check if it's a JSON file path
            if transformer_namespace.endswith('.json'):
                if os.path.isfile(transformer_namespace):
                    json_transformers = self._load_transformers_from_json(transformer_namespace)
                    self.transformer_collection = {
                        **self.transformer_collection,
                        **json_transformers,
                    }
                else:
                    raise FileNotFoundError(f"Transformer JSON file not found: {transformer_namespace}")
            else:
                # Handle existing preset names
                transformer_subspace = transformer_vault._tr_global_namespace.get(
                    transformer_namespace, None,
                )
                if transformer_subspace:
                    self.transformer_collection = {
                        **self.transformer_collection,
                        **transformer_subspace,
                    }

        if len(self.transformer_collection) == 0:
            raise NotImplementedError(
                'Please, specify valid transformer namespaces (e.g., default, minimal etc.) or provide a valid JSON file path.',
            )

        self.numeric_column_names = set(numeric_column_names)
        self.constructed_feature_names: set[str] = set()

        # If 80% of values are the same, don't consider a transformation
        self.max_maj_support = 0.80

        # If more than 75% of vals are missing, don't consider a transformation
        self.nan_prop_support = 0.75

    def _load_transformers_from_json(self, json_file_path: str) -> dict[str, str]:
        """Load transformer specifications from a JSON file."""
        try:
            with open(json_file_path, 'r') as f:
                transformers = json.load(f)
            
            if not isinstance(transformers, dict):
                raise ValueError(f"JSON file {json_file_path} must contain a dictionary of transformer specifications")
            
            # Validate that all values are strings (transformer expressions)
            for key, value in transformers.items():
                if not isinstance(value, str):
                    raise ValueError(f"Transformer '{key}' in {json_file_path} must have a string expression, got {type(value)}")
            
            logging.info(f"Loaded {len(transformers)} transformers from {json_file_path}")
            return transformers
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in transformer file {json_file_path}: {e}")
        except Exception as e:
            raise

    def get_vals(self, tmp_df: pd.DataFrame, col_name: str) -> Any:
        cvals = tmp_df[col_name].values.tolist()
        cvals = [str(x).replace('"', '') for x in cvals]
        cvals = [0.0 if len(x) == 0 else float(x) for x in cvals]

        return np.array(cvals)

    def construct_baseline_features(self, dataframe: Any) -> pd.DataFrame:
        fvals = []
        for enx, row in dataframe.iterrows():
            missing_prop = np.round(
                row.values.tolist().count('') / dataframe.shape[1], 1,
            )
            fvals.append(missing_prop)

        dataframe['BASELINE-MISSING-PROPORTION'] = fvals
        dataframe['BASELINE-DUMMY'] = 0

        return dataframe

    def construct_new_features(self, dataframe: Any) -> pd.DataFrame:
        new_numeric = set()
        logging.info(
            f'Considering {len(self.transformer_collection)} transformations for {len(self.numeric_column_names)} features ({len(self.transformer_collection) * len(self.numeric_column_names)} new features will be considered).',
        )

        invalid_transforms = 0
        new_columns = dict()
        for numeric_column in self.numeric_column_names:
            X = self.get_vals(dataframe, numeric_column)

            if len(X) == 0:
                raise AssertionError(
                    f"Could not retrieve the colomn {numeric_column}'s values. Please check the data.",
                )

            for k, v in self.transformer_collection.items():
                feature_name = f'{numeric_column}{k}'
                transformed_array = eval(v).astype(str)
                u, c = np.unique(transformed_array, return_counts=True)
                nan_prop = np.count_nonzero(transformed_array == 'nan') / len(
                    transformed_array,
                )
                cfreq = np.divide(np.max(c), np.sum(c))
                if (
                    len(u) > 1
                    and cfreq < self.max_maj_support
                    and nan_prop < self.nan_prop_support
                ):
                    new_columns[feature_name] = transformed_array
                    new_numeric.add(feature_name)

                else:
                    invalid_transforms += 1

        if len(new_columns) > 0:
            tmp_df = pd.DataFrame(new_columns)
            dataframe = pd.concat([dataframe, tmp_df], axis=1)
            del tmp_df

        logging.info(
            f'{invalid_transforms} invalid transformations were skipped.',
        )
        self.numeric_column_names = self.numeric_column_names
        self.constructed_feature_names = new_numeric
        return dataframe
