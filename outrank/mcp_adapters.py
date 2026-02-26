"""Adapter layer for the OutRank MCP server.

Bridges MCP tool calls to OutRank internals:
- Namespace builder (avoids importing __main__ argparse)
- Logging redirect (stdout -> stderr, critical for stdio JSON-RPC)
- SystemExit guard (task_ranking.py calls exit() in several places)
- DataFrame -> JSON conversion
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Any


def configure_logging_for_mcp() -> None:
    """Redirect ALL logging output to stderr.

    OutRank modules call logging.basicConfig() at import time
    (__main__.py:13, task_ranking.py:24, core_utils.py:17,
    task_generators.py:12). On stdio MCP transport, any stdout
    output corrupts the JSON-RPC framing. This must be called
    BEFORE importing any outrank module.
    """
    root = logging.getLogger()
    # Remove any existing handlers (including stdout ones
    # that basicConfig may have installed)
    for handler in root.handlers[:]:
        root.removeHandler(handler)
    # Install a single stderr handler
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setFormatter(
        logging.Formatter('%(asctime)s - %(name)s - %(message)s'),
    )
    root.addHandler(stderr_handler)
    root.setLevel(logging.INFO)

    # Monkey-patch basicConfig to be a no-op so subsequent OutRank
    # imports don't re-add stdout handlers.
    logging.basicConfig = lambda **kwargs: None  # type: ignore[assignment]


def build_ranking_namespace(**overrides: Any) -> argparse.Namespace:
    """Build an argparse.Namespace matching all CLI flags.

    Defaults are hardcoded here (mirrored from __main__.py) to avoid
    importing __main__ and triggering its logging.basicConfig().
    """
    defaults = {
        'task': 'ranking',
        'minibatch_size': 2**14,
        'output_folder': '/tmp/outrank_mcp_output',
        'data_source': 'csv-raw',
        'data_path': None,
        'subsampling': 10,
        'combination_number_upper_bound': 2**15,
        'missing_value_symbols': ',{}',
        'heuristic': 'MI-numba-randomized',
        'include_noise_baseline_features': 'False',
        'include_cardinality_in_feature_names': 'True',
        'image_format': 'pdf',
        'num_threads': 4,
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
        'tldr': 'False',
        'num_synthetic_rows': 10000,
        'generator_type': 'naive',
        'output_synthetic_df_name': 'test_data_synthetic',
        'disable_tqdm': 'True',  # Always suppress in MCP context
        'mi_stratified_sampling_ratio': 1.0,
        'compute_jmi': 'False',
        'jmi_top_k': 50,
        'compute_interaction_info': 'False',
        'interaction_info_top_k': 30,
    }
    # Apply caller overrides, converting Python bools to OutRank string convention
    for key, value in overrides.items():
        if isinstance(value, bool):
            value = 'True' if value else 'False'
        defaults[key] = value

    return argparse.Namespace(**defaults)


def run_ranking_safe(args: argparse.Namespace) -> dict[str, Any]:
    """Run outrank_task_conduct_ranking with SystemExit protection.

    task_ranking.py calls exit() at lines 130, 139, 163.
    os.remove('ranking_checkpoint_tmp.tsv') at line 317 may fail.
    Returns a dict with parsed output files.
    """
    import time

    from outrank.task_ranking import outrank_task_conduct_ranking

    start = time.monotonic()
    error_msg = None
    try:
        outrank_task_conduct_ranking(args)
    except SystemExit:
        # Lines 130, 139, 163 — non-fatal for our purposes
        pass
    except Exception as exc:
        error_msg = f'{type(exc).__name__}: {exc}'
        logging.getLogger('outrank.mcp').warning(
            'Ranking failed: %s', error_msg,
        )
    elapsed = time.monotonic() - start

    result: dict[str, Any] = {
        'output_folder': args.output_folder,
        'elapsed_seconds': round(elapsed, 2),
    }
    if error_msg is not None:
        result['error'] = error_msg

    # Read back pairwise ranks if they were written
    pairwise_path = os.path.join(args.output_folder, 'pairwise_ranks.tsv')
    if os.path.exists(pairwise_path):
        import pandas as pd
        df = pd.read_csv(pairwise_path, sep='\t')
        result['pairwise_ranks'] = dataframe_to_ranking_json(df)

    # JMI results
    jmi_path = os.path.join(args.output_folder, 'jmi_feature_ranking.tsv')
    if os.path.exists(jmi_path):
        import pandas as pd
        df = pd.read_csv(jmi_path, sep='\t')
        result['jmi_ranking'] = df.to_dict(orient='records')

    # Interaction information
    ii_path = os.path.join(args.output_folder, 'interaction_information.tsv')
    if os.path.exists(ii_path):
        import pandas as pd
        df = pd.read_csv(ii_path, sep='\t')
        result['interaction_info'] = df.to_dict(orient='records')

    # Timings
    timings_path = os.path.join(args.output_folder, 'timings.json')
    if os.path.exists(timings_path):
        import json
        with open(timings_path) as f:
            result['timings'] = json.load(f)

    return result


def dataframe_to_ranking_json(
    df: Any, top_n: int | None = None,
) -> list[dict[str, Any]]:
    """Convert a pairwise_ranks DataFrame to JSON-serializable list."""
    if top_n is not None:
        df = df.head(top_n)
    records = []
    for _, row in df.iterrows():
        records.append({
            'feature_a': str(row.iloc[0]),
            'feature_b': str(row.iloc[1]),
            'score': float(row.iloc[2]),
        })
    return records
