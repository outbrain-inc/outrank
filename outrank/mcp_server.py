"""OutRank MCP Server — exposes feature ranking as callable tools.

Transport: stdio (standard for Claude Code).
Start:     python -m outrank.mcp_server

CRITICAL: configure_logging_for_mcp() MUST run before any outrank import
to prevent stdout pollution that would corrupt JSON-RPC framing.
"""
from __future__ import annotations

from outrank.mcp_adapters import configure_logging_for_mcp
# --- Logging redirect (MUST be first) ---
configure_logging_for_mcp()

import json
import logging
import os
import sys
from typing import Any

import numpy as np
import pandas as pd

from mcp.server.fastmcp import FastMCP

from outrank.mcp_adapters import (
    build_ranking_namespace,
    dataframe_to_ranking_json,
    run_ranking_safe,
)

logger = logging.getLogger('outrank.mcp')

# ---------------------------------------------------------------------------
# Server instance
# ---------------------------------------------------------------------------
mcp = FastMCP(
    'outrank',
    instructions='OutRank: fast feature ranking for sparse data sets. '
    'Provides mutual-information-based feature scoring, '
    'JMI greedy selection, interaction information, '
    'and synthetic data generation.',
)

# ---------------------------------------------------------------------------
# Heuristic registry (static, no computation)
# ---------------------------------------------------------------------------
HEURISTICS = [
    {
        'name': 'MI-numba-randomized',
        'description': 'Mutual information with cardinality correction (Numba JIT). '
        'Prevents high-cardinality features from inflating scores.',
        'speed': 'fast',
        'recommended_for': 'Default choice. Best balance of speed and accuracy.',
    },
    {
        'name': 'MI-numba-randomized-opt',
        'description': 'Optimized MI-numba variant with identical semantics but faster inner loop.',
        'speed': 'fast',
        'recommended_for': 'Same as MI-numba-randomized; slightly faster on large cardinalities.',
    },
    {
        'name': 'MI',
        'description': 'Mutual information via sklearn (mutual_info_classif). No cardinality correction.',
        'speed': 'moderate',
        'recommended_for': 'Baseline comparison or when Numba is unavailable.',
    },
    {
        'name': 'AMI',
        'description': 'Adjusted mutual information (sklearn). Corrects for chance agreement.',
        'speed': 'moderate',
        'recommended_for': 'When features have very different cardinalities and you want chance-corrected scores.',
    },
    {
        'name': 'correlation-Pearson',
        'description': 'Pearson correlation coefficient. Linear relationships only.',
        'speed': 'fast',
        'recommended_for': 'Numeric features with suspected linear relationship to label.',
    },
    {
        'name': 'surrogate-SGD',
        'description': 'SGD logistic regression surrogate. Cross-validated log-loss.',
        'speed': 'slow',
        'recommended_for': 'When you want model-based feature importance (linear).',
    },
    {
        'name': 'surrogate-SGD-SVD',
        'description': 'SGD with SVD dimensionality reduction. Handles high-cardinality OHE.',
        'speed': 'slow',
        'recommended_for': 'High-cardinality features where full OHE is too wide.',
    },
    {
        'name': 'surrogate-SGD-RP',
        'description': 'SGD with random projection. Faster than SVD for very wide features.',
        'speed': 'slow',
        'recommended_for': 'Very high-cardinality features.',
    },
    {
        'name': 'surrogate-SVM',
        'description': 'SVM surrogate with RBF kernel. Cross-validated log-loss.',
        'speed': 'very slow',
        'recommended_for': 'Non-linear feature importance (small datasets only).',
    },
    {
        'name': 'max-value-coverage',
        'description': 'Coverage alignment heuristic. Scores by value co-occurrence.',
        'speed': 'fast',
        'recommended_for': 'Exploratory analysis of feature coverage patterns.',
    },
    {
        'name': 'MI-multivalue-jaccard',
        'description': 'MI for semicolon-separated multivalue features using Jaccard similarity.',
        'speed': 'moderate',
        'recommended_for': 'Multivalue features (e.g., tags, categories) with Jaccard distance.',
    },
    {
        'name': 'MI-multivalue-overlap',
        'description': 'MI for multivalue features using overlap coefficient.',
        'speed': 'moderate',
        'recommended_for': 'Multivalue features where subset relationships matter.',
    },
    {
        'name': 'MI-multivalue-set',
        'description': 'MI for multivalue features using set-based encoding.',
        'speed': 'moderate',
        'recommended_for': 'Multivalue features with set-based similarity.',
    },
    {
        'name': 'MI-multivalue-set-randomized',
        'description': 'Set-based MI with cardinality correction for multivalue features.',
        'speed': 'moderate',
        'recommended_for': 'Multivalue features where cardinality varies widely.',
    },
    {
        'name': 'Constant',
        'description': 'Returns 0.0 for all pairs. Used internally for rare-value and transformer tasks.',
        'speed': 'instant',
        'recommended_for': 'Internal use only (feature_summary_transformers, identify_rare_values).',
    },
]

USAGE_GUIDE = """\
# OutRank MCP Usage Guide

## Data Format
OutRank expects CSV files with a header row. The label/target column
defaults to "label" but can be set via `label_column`.

For `data_source="csv-raw"`, provide the path to a directory containing
a single `data.csv` file, or point `data_path` directly at a `.csv` file.

## Common Workflows

### 1. Quick feature ranking (target-only)
Use `outrank_rank_features` with `target_ranking_only=True` (default).
This scores each feature against the label — O(n) pairs, fast.

### 2. Full pairwise ranking
Set `target_ranking_only=False` to score all feature pairs — O(n^2).
Use `subsampling` to control speed vs. accuracy.

### 3. Synergy detection
Enable `compute_jmi=True` for JMI greedy selection (finds XOR-like
synergies) and/or `compute_interaction_info=True` for pairwise
interaction information (negative = synergy, positive = redundancy).

### 4. Quick pair check
Use `outrank_score_pair` to score two specific features without
running the full pipeline.

### 5. Dataset inspection
Use `outrank_dataset_info` before ranking to verify column names,
cardinalities, and data shape.

## Heuristic Selection
- **MI-numba-randomized** (default): Fast, accurate, handles cardinality.
- **MI**: Sklearn baseline, no cardinality correction.
- **surrogate-***: Model-based importance. Slower but captures non-linear signal.
- **MI-multivalue-***: For semicolon-separated multivalue features.

## Performance Tips
- `subsampling=100` → 10x faster, usually sufficient for ranking order.
- `num_threads=8` → Good default for most machines.
- `target_ranking_only=True` → O(n) instead of O(n^2).
- `combination_number_upper_bound` → Caps pairs per batch (Monte Carlo sampling).

## Gotchas
- Boolean flags use string values: `"True"` / `"False"` (not Python bools).
  The MCP tools handle this conversion automatically.
- The label column must exist in the data.
- For `csv-raw` data_source, `data_path` should point to a directory
  containing `data.csv`, or to a csv file directly.
"""

# ---------------------------------------------------------------------------
# Resources
# ---------------------------------------------------------------------------

@mcp.resource('outrank://heuristics')
def resource_heuristics() -> str:
    """List of all available scoring heuristics with descriptions."""
    return json.dumps(HEURISTICS, indent=2)


@mcp.resource('outrank://guide')
def resource_guide() -> str:
    """Condensed usage guide for agents — data formats, workflows, gotchas."""
    return USAGE_GUIDE


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@mcp.tool()
def outrank_list_heuristics() -> str:
    """List all available scoring heuristics with descriptions, speed ratings, and recommendations.

    No parameters needed. Returns a JSON array of heuristic objects.
    """
    return json.dumps(HEURISTICS, indent=2)


@mcp.tool()
def outrank_dataset_info(
    data_path: str,
    data_source: str = 'csv-raw',
    sample_rows: int = 5,
) -> str:
    """Inspect a dataset's structure before ranking.

    Returns column names, row count, cardinalities, and a sample of data.
    Use this to verify the dataset is correctly formatted before running
    a full ranking.

    Args:
        data_path: Path to the data directory or CSV file.
        data_source: Data source format. Use "csv-raw" for plain CSV files.
        sample_rows: Number of sample rows to include in the response.
    """
    csv_path = _resolve_csv_path(data_path, data_source)
    df = pd.read_csv(csv_path, nrows=sample_rows + 1)
    df_full_shape = pd.read_csv(csv_path, usecols=[0])
    num_rows = len(df_full_shape)

    # Re-read sample with all columns
    df_sample = pd.read_csv(csv_path, nrows=sample_rows)
    cardinalities = {}
    # Read more rows for cardinality estimation
    df_card = pd.read_csv(csv_path, nrows=min(10000, num_rows))
    for col in df_card.columns:
        cardinalities[col] = int(df_card[col].nunique())

    result = {
        'columns': list(df_sample.columns),
        'num_rows': num_rows,
        'num_columns': len(df_sample.columns),
        'column_cardinalities': cardinalities,
        'sample_data': df_sample.to_dict(orient='records'),
    }
    return json.dumps(result, indent=2, default=str)


@mcp.tool()
def outrank_rank_features(
    data_path: str,
    label_column: str = 'label',
    heuristic: str = 'MI-numba-randomized',
    data_source: str = 'csv-raw',
    target_ranking_only: bool = True,
    subsampling: int = 10,
    num_threads: int = 4,
    compute_jmi: bool = False,
    jmi_top_k: int = 50,
    compute_interaction_info: bool = False,
    interaction_info_top_k: int = 30,
    output_folder: str = '/tmp/outrank_mcp_output',
) -> str:
    """Run the full OutRank feature ranking pipeline on a dataset.

    This is the primary tool. It scores features against a label column
    using mutual-information-based heuristics. Optionally computes JMI
    (Joint Mutual Information) greedy selection and interaction information.

    Args:
        data_path: Path to the data directory containing data.csv, or path to a CSV file.
        label_column: Name of the target/label column.
        heuristic: Scoring heuristic. Use outrank_list_heuristics to see all options.
        data_source: Data source format. Use "csv-raw" for plain CSV.
        target_ranking_only: If True, only score features vs label (fast, O(n)). If False, score all pairs (O(n^2)).
        subsampling: Subsample ratio — every n-th instance. Higher = faster but less precise.
        num_threads: Number of parallel threads.
        compute_jmi: If True, run JMI greedy selection after pairwise ranking. Detects synergies.
        jmi_top_k: Number of top features to consider for JMI.
        compute_interaction_info: If True, compute interaction information for top feature pairs.
        interaction_info_top_k: Number of top features for interaction info computation.
        output_folder: Where to write output files.
    """
    args = build_ranking_namespace(
        data_path=data_path,
        label_column=label_column,
        heuristic=heuristic,
        data_source=data_source,
        target_ranking_only=target_ranking_only,
        subsampling=subsampling,
        num_threads=num_threads,
        compute_jmi=compute_jmi,
        jmi_top_k=jmi_top_k,
        compute_interaction_info=compute_interaction_info,
        interaction_info_top_k=interaction_info_top_k,
        output_folder=output_folder,
    )
    result = run_ranking_safe(args)
    return json.dumps(result, indent=2, default=str)


@mcp.tool()
def outrank_score_pair(
    data_path: str,
    feature_a: str,
    feature_b: str,
    heuristic: str = 'MI-numba-randomized',
    data_source: str = 'csv-raw',
    label_column: str = 'label',
    subsampling: int = 1,
) -> str:
    """Score two specific features without running the full pipeline.

    Reads the CSV directly, encodes the two columns, and computes the
    pairwise score. Much faster than outrank_rank_features for quick checks.

    Args:
        data_path: Path to the data directory or CSV file.
        feature_a: Name of the first feature column.
        feature_b: Name of the second feature column (often the label).
        heuristic: Scoring heuristic to use.
        data_source: Data source format.
        label_column: Name of the label column (used for args construction).
        subsampling: Subsample ratio. 1 = use all rows.
    """
    from outrank.algorithms.importance_estimator import conduct_feature_ranking

    csv_path = _resolve_csv_path(data_path, data_source)
    df = pd.read_csv(csv_path, usecols=[feature_a, feature_b])
    if subsampling > 1:
        df = df.iloc[::subsampling]

    v1, _ = pd.factorize(df[feature_a])
    v2, _ = pd.factorize(df[feature_b])

    v1 = v1.astype(np.int32)
    v2 = v2.astype(np.int32)

    args = build_ranking_namespace(
        heuristic=heuristic,
        label_column=label_column,
        mi_stratified_sampling_ratio=1.0,
    )
    score = conduct_feature_ranking(v1, v2, args)

    result = {
        'feature_a': feature_a,
        'feature_b': feature_b,
        'score': float(score),
        'heuristic': heuristic,
        'n_samples': len(df),
    }
    return json.dumps(result, indent=2)


@mcp.tool()
def outrank_compute_interaction_info(
    data_path: str,
    feature_a: str,
    feature_b: str,
    label_column: str = 'label',
    data_source: str = 'csv-raw',
    cardinality_correction: bool = False,
) -> str:
    """Compute interaction information II(feature_a, feature_b; label).

    Measures synergy vs. redundancy between two features with respect to the label.
    Uses the Jakulin & Bratko convention: negative = synergy, positive = redundancy.

    Synergy means the features are more informative together than separately.
    Redundancy means they carry overlapping information about the label.

    Args:
        data_path: Path to the data directory or CSV file.
        feature_a: First feature column name.
        feature_b: Second feature column name.
        label_column: Target/label column name.
        data_source: Data source format.
        cardinality_correction: Apply cardinality correction to MI estimates.
    """
    from outrank.algorithms.feature_ranking import ranking_mi_numba_cmi

    csv_path = _resolve_csv_path(data_path, data_source)
    df = pd.read_csv(csv_path, usecols=[feature_a, feature_b, label_column])

    y, _ = pd.factorize(df[label_column])
    x1, _ = pd.factorize(df[feature_a])
    x2, _ = pd.factorize(df[feature_b])

    y = y.astype(np.int32)
    x1 = x1.astype(np.int32)
    x2 = x2.astype(np.int32)

    ii = float(
        ranking_mi_numba_cmi.interaction_information_numba(
            y, x1, x2, np.float32(1.0), cardinality_correction,
        ),
    )

    if ii < -0.01:
        interpretation = 'synergy (features are more informative together than separately)'
    elif ii > 0.01:
        interpretation = 'redundancy (features carry overlapping information about the label)'
    else:
        interpretation = 'near-zero (features contribute roughly independently)'

    result = {
        'feature_a': feature_a,
        'feature_b': feature_b,
        'interaction_info': ii,
        'interpretation': interpretation,
        'convention': 'Jakulin & Bratko (negative=synergy, positive=redundancy)',
    }
    return json.dumps(result, indent=2)


@mcp.tool()
def outrank_generate_synthetic_data(
    output_path: str = '/tmp/outrank_synthetic',
    num_features: int = 100,
    num_rows: int = 10000,
    generator_type: str = 'naive',
) -> str:
    """Generate a synthetic dataset for testing OutRank.

    Creates a CSV with the specified number of features and rows,
    including a "label" column. Useful for testing ranking workflows
    without requiring real data.

    Args:
        output_path: Directory to write the generated data.csv.
        num_features: Number of features to generate.
        num_rows: Number of rows to generate.
        generator_type: Generator type. Currently only "naive" is supported.
    """
    from outrank.algorithms.synthetic_data_generators import generator_naive

    if generator_type != 'naive':
        return json.dumps({'error': f'Generator {generator_type} not implemented.'})

    # Call generator directly to avoid task_generators.py's `./` path prefix bug
    sample, target = generator_naive.generate_random_matrix(num_features, num_rows)
    df = pd.DataFrame(sample, columns=[f'f{i}' for i in range(num_features)])
    df['label'] = target

    os.makedirs(output_path, exist_ok=True)
    df.to_csv(os.path.join(output_path, 'data.csv'), index=False)

    csv_path = os.path.join(output_path, 'data.csv')
    actual_shape = None
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, nrows=1)
        # Get row count cheaply
        with open(csv_path) as f:
            actual_rows = sum(1 for _ in f) - 1  # minus header
        actual_shape = {'rows': actual_rows, 'columns': len(df.columns)}

    result = {
        'output_path': output_path,
        'csv_file': csv_path,
        'requested': {'num_features': num_features, 'num_rows': num_rows},
        'actual_shape': actual_shape,
    }
    return json.dumps(result, indent=2)


@mcp.tool()
def outrank_summarize_results(
    output_folder: str,
    top_n: int = 20,
) -> str:
    """Read and summarize ranking output from a previous OutRank run.

    Parses pairwise_ranks.tsv and returns the top-N features sorted by score.
    Also includes JMI and interaction info results if available.

    Args:
        output_folder: Path to the OutRank output folder (containing pairwise_ranks.tsv).
        top_n: Number of top results to return.
    """
    result: dict[str, Any] = {}

    pairwise_path = os.path.join(output_folder, 'pairwise_ranks.tsv')
    if os.path.exists(pairwise_path):
        df = pd.read_csv(pairwise_path, sep='\t')
        df = df.sort_values(by=df.columns[2], ascending=False)
        top = df.head(top_n)
        result['top_pairwise_ranks'] = dataframe_to_ranking_json(top)

        # Build markdown table
        lines = ['| Rank | Feature A | Feature B | Score |', '|------|-----------|-----------|-------|']
        for i, (_, row) in enumerate(top.iterrows(), 1):
            lines.append(f'| {i} | {row.iloc[0]} | {row.iloc[1]} | {row.iloc[2]:.6f} |')
        result['markdown_table'] = '\n'.join(lines)
    else:
        result['error'] = f'pairwise_ranks.tsv not found in {output_folder}'

    jmi_path = os.path.join(output_folder, 'jmi_feature_ranking.tsv')
    if os.path.exists(jmi_path):
        jmi_df = pd.read_csv(jmi_path, sep='\t')
        result['jmi_ranking'] = jmi_df.to_dict(orient='records')

    ii_path = os.path.join(output_folder, 'interaction_information.tsv')
    if os.path.exists(ii_path):
        ii_df = pd.read_csv(ii_path, sep='\t')
        ii_df = ii_df.sort_values(by='InteractionInfo')
        result['interaction_info'] = ii_df.head(top_n).to_dict(orient='records')

    return json.dumps(result, indent=2, default=str)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_csv_path(data_path: str, data_source: str) -> str:
    """Resolve a data_path to the actual CSV file path.

    For csv-raw, data_path may be:
    - A directory containing data.csv
    - A direct path to a .csv file
    """
    if os.path.isfile(data_path) and data_path.endswith('.csv'):
        return data_path
    csv_in_dir = os.path.join(data_path, 'data.csv')
    if os.path.isfile(csv_in_dir):
        return csv_in_dir
    raise FileNotFoundError(
        f'Could not find CSV data at {data_path}. '
        f'Expected a .csv file or a directory containing data.csv.',
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    mcp.run(transport='stdio')


if __name__ == '__main__':
    main()
