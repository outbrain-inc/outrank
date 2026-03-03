"""Single source of truth for CLI/MCP default values.

Both __main__.py (argparse) and mcp_adapters.py (Namespace builder)
import from here to prevent silent divergence of defaults.
"""
from __future__ import annotations

CLI_DEFAULTS: dict[str, object] = {
    'task': 'all',
    'minibatch_size': 2**14,
    'output_folder': 'ranking_outputs',
    'data_source': 'ob-vw',
    'data_path': None,
    'subsampling': 10,
    'combination_number_upper_bound': 2**15,
    'missing_value_symbols': ',{}',
    'heuristic': 'MI-numba-randomized',
    'include_noise_baseline_features': 'False',
    'include_cardinality_in_feature_names': 'True',
    'image_format': 'pdf',
    'num_threads': 8,
    'label_column': 'label',
    'max_unique_hist_constraint': 30_000,
    'transformers': 'none',
    'rare_value_count_upper_bound': 1,
    'feature_set_focus': None,
    'interaction_order': 1,
    'reference_model_JSON': '',
    'target_ranking_only': 'True',
    'explode_multivalue_features': 'False',
    'subfeature_mapping': 'False',
    'num_synthetic_features': 100,
    'tldr': 'True',
    'num_synthetic_rows': 1000000,
    'generator_type': 'naive',
    'output_synthetic_df_name': 'test_data_synthetic',
    'disable_tqdm': 'False',
    'mi_stratified_sampling_ratio': 1.0,
    'compute_jmi': 'False',
    'jmi_top_k': 50,
    'compute_interaction_info': 'False',
    'interaction_info_top_k': 30,
}
