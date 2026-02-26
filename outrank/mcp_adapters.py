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


MAX_FILE_SIZE_MB = 500


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

    Defaults come from cli_defaults.py (single source of truth),
    with MCP-specific overrides applied on top.
    """
    from outrank.cli_defaults import CLI_DEFAULTS

    defaults = dict(CLI_DEFAULTS)
    # MCP-specific overrides: headless mode
    defaults['task'] = 'ranking'
    defaults['output_folder'] = '/tmp/outrank_mcp_output'
    defaults['data_source'] = 'csv-raw'
    defaults['num_threads'] = 4
    defaults['disable_tqdm'] = 'True'
    defaults['tldr'] = 'False'
    defaults['num_synthetic_rows'] = 10000
    # Apply caller overrides, converting Python bools to OutRank string convention
    for key, value in overrides.items():
        if isinstance(value, bool):
            value = 'True' if value else 'False'
        defaults[key] = value

    return argparse.Namespace(**defaults)


def validate_data_path(data_path: str | None) -> None:
    """Validate data_path exists and is accessible. Raises ValueError on failure."""
    if data_path is None:
        raise ValueError('data_path must not be None')
    if not os.path.exists(data_path):
        raise ValueError(f"data_path '{data_path}' does not exist")
    if os.path.isfile(data_path):
        size_mb = os.path.getsize(data_path) / (1024 * 1024)
        if size_mb > MAX_FILE_SIZE_MB:
            raise ValueError(
                f"File '{data_path}' is {size_mb:.0f} MB, "
                f'exceeding the {MAX_FILE_SIZE_MB} MB limit',
            )
    elif os.path.isdir(data_path):
        csv_path = os.path.join(data_path, 'data.csv')
        if os.path.isfile(csv_path):
            size_mb = os.path.getsize(csv_path) / (1024 * 1024)
            if size_mb > MAX_FILE_SIZE_MB:
                raise ValueError(
                    f"File '{csv_path}' is {size_mb:.0f} MB, "
                    f'exceeding the {MAX_FILE_SIZE_MB} MB limit',
                )


def run_ranking_safe(args: argparse.Namespace) -> dict[str, Any]:
    """Run outrank_task_conduct_ranking with SystemExit protection.

    task_ranking.py calls exit() at lines 130, 139, 163.
    os.remove('ranking_checkpoint_tmp.tsv') at line 317 may fail.
    Returns a dict with parsed output files.
    """
    import time

    from outrank.task_ranking import outrank_task_conduct_ranking

    # Validate inputs before launching the ranking pipeline
    try:
        validate_data_path(args.data_path)
    except ValueError as exc:
        return {
            'output_folder': args.output_folder,
            'elapsed_seconds': 0,
            'error': str(exc),
        }

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
