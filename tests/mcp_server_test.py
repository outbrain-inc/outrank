"""Tests for OutRank MCP server tools.

All tests call tool functions directly as regular Python functions — no MCP
transport needed. The functions return JSON strings which we parse and assert on.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import sys
import tempfile

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope='session')
def synthetic_data_dir():
    """Generate a small synthetic dataset for tests. Reused across the session."""
    tmpdir = tempfile.mkdtemp(prefix='outrank_mcp_test_')
    rng = np.random.RandomState(42)
    n = 2000
    n_features = 30

    # f0-f29: random categorical features with varying cardinality
    data = {}
    for i in range(n_features):
        card = rng.randint(2, 20)
        data[f'f{i}'] = rng.randint(0, card, size=n)

    # f30: correlated with label (strong signal)
    label = rng.randint(0, 3, size=n)
    data['f30'] = label.copy()
    # Add some noise to f30
    flip_mask = rng.random(n) < 0.1
    data['f30'][flip_mask] = rng.randint(0, 3, size=flip_mask.sum())

    data['label'] = label

    df = pd.DataFrame(data)
    csv_dir = os.path.join(tmpdir, 'data_dir')
    os.makedirs(csv_dir)
    df.to_csv(os.path.join(csv_dir, 'data.csv'), index=False)

    yield csv_dir

    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture(scope='session')
def xor_data_dir():
    """Dataset with XOR-like synergy: Y = (X1 + X2) % 3."""
    tmpdir = tempfile.mkdtemp(prefix='outrank_mcp_xor_')
    rng = np.random.RandomState(123)
    n = 5000
    x1 = rng.randint(0, 3, size=n)
    x2 = rng.randint(0, 3, size=n)
    y = (x1 + x2) % 3

    df = pd.DataFrame({'x1': x1, 'x2': x2, 'noise': rng.randint(0, 5, size=n), 'label': y})
    csv_dir = os.path.join(tmpdir, 'xor_dir')
    os.makedirs(csv_dir)
    df.to_csv(os.path.join(csv_dir, 'data.csv'), index=False)

    yield csv_dir

    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture
def output_dir():
    tmpdir = tempfile.mkdtemp(prefix='outrank_mcp_out_')
    yield tmpdir
    shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Adapter tests
# ---------------------------------------------------------------------------

class TestAdapters:
    def test_build_namespace_defaults(self):
        from outrank.mcp_adapters import build_ranking_namespace

        ns = build_ranking_namespace()
        # Verify key defaults match __main__.py
        assert ns.task == 'ranking'
        assert ns.heuristic == 'MI-numba-randomized'
        assert ns.subsampling == 10
        assert ns.label_column == 'label'
        assert ns.data_source == 'csv-raw'
        assert ns.target_ranking_only == 'True'
        assert ns.disable_tqdm == 'True'
        assert ns.compute_jmi == 'False'
        assert ns.compute_interaction_info == 'False'
        assert ns.num_threads == 4
        assert ns.minibatch_size == 2**14
        assert ns.combination_number_upper_bound == 2**15

    def test_bool_string_conversion(self):
        from outrank.mcp_adapters import build_ranking_namespace

        ns = build_ranking_namespace(
            target_ranking_only=True,
            compute_jmi=False,
            compute_interaction_info=True,
        )
        assert ns.target_ranking_only == 'True'
        assert ns.compute_jmi == 'False'
        assert ns.compute_interaction_info == 'True'

    def test_overrides_applied(self):
        from outrank.mcp_adapters import build_ranking_namespace

        ns = build_ranking_namespace(
            data_path='/some/path',
            heuristic='AMI',
            num_threads=16,
            subsampling=50,
        )
        assert ns.data_path == '/some/path'
        assert ns.heuristic == 'AMI'
        assert ns.num_threads == 16
        assert ns.subsampling == 50

    def test_logging_stderr_only(self):
        """After configure_logging_for_mcp, no handlers should write to stdout."""
        from outrank.mcp_adapters import configure_logging_for_mcp

        configure_logging_for_mcp()
        root = logging.getLogger()
        for handler in root.handlers:
            if isinstance(handler, logging.StreamHandler):
                assert handler.stream is not sys.stdout, \
                    'Found a stdout handler after configure_logging_for_mcp()'

    def test_system_exit_handled(self):
        """run_ranking_safe should catch SystemExit and other exceptions without propagating."""
        from outrank.mcp_adapters import build_ranking_namespace, run_ranking_safe

        # Pass an invalid data_path that will cause task_ranking to fail
        args = build_ranking_namespace(data_path='/nonexistent/path/definitely')
        # This should NOT raise — errors are captured in the result dict
        result = run_ranking_safe(args)
        assert isinstance(result, dict)
        assert 'output_folder' in result
        assert 'error' in result  # Should report the error

    def test_dataframe_to_ranking_json(self):
        from outrank.mcp_adapters import dataframe_to_ranking_json

        df = pd.DataFrame({
            'FeatureA': ['f1', 'f2', 'f3'],
            'FeatureB': ['label', 'label', 'label'],
            'Score': [0.5, 0.3, 0.1],
        })
        records = dataframe_to_ranking_json(df, top_n=2)
        assert len(records) == 2
        assert records[0]['feature_a'] == 'f1'
        assert records[0]['score'] == 0.5


# ---------------------------------------------------------------------------
# Tool tests
# ---------------------------------------------------------------------------

class TestListHeuristics:
    def test_returns_all_heuristics(self):
        from outrank.mcp_server import outrank_list_heuristics

        result = json.loads(outrank_list_heuristics())
        assert isinstance(result, list)
        assert len(result) >= 14
        names = {h['name'] for h in result}
        assert 'MI-numba-randomized' in names
        assert 'AMI' in names
        assert 'Constant' in names

    def test_heuristic_fields(self):
        from outrank.mcp_server import outrank_list_heuristics

        result = json.loads(outrank_list_heuristics())
        for h in result:
            assert 'name' in h
            assert 'description' in h
            assert 'speed' in h
            assert 'recommended_for' in h


class TestDatasetInfo:
    def test_dataset_info(self, synthetic_data_dir):
        from outrank.mcp_server import outrank_dataset_info

        result = json.loads(
            outrank_dataset_info(
                data_path=synthetic_data_dir,
                sample_rows=3,
            ),
        )
        assert 'label' in result['columns']
        assert result['num_rows'] == 2000
        assert result['num_columns'] == 32  # f0-f30 + label
        assert len(result['sample_data']) == 3
        assert 'label' in result['column_cardinalities']


class TestGenerateSyntheticData:
    def test_generate(self, output_dir):
        from outrank.mcp_server import outrank_generate_synthetic_data

        out_path = os.path.join(output_dir, 'gen_data')
        # generator_naive uses sample[:, 30] as target, so need >= 31 features
        result = json.loads(
            outrank_generate_synthetic_data(
                output_path=out_path,
                num_features=50,
                num_rows=500,
            ),
        )
        assert os.path.exists(result['csv_file'])
        assert result['actual_shape'] is not None
        assert result['actual_shape']['rows'] == 500
        assert result['actual_shape']['columns'] == 51  # 50 features + label


class TestScorePair:
    def test_score_pair_signal_vs_noise(self, synthetic_data_dir):
        """f30 (correlated with label) should score higher than f0 (random)."""
        from outrank.mcp_server import outrank_score_pair

        signal = json.loads(
            outrank_score_pair(
                data_path=synthetic_data_dir,
                feature_a='f30',
                feature_b='label',
            ),
        )
        noise = json.loads(
            outrank_score_pair(
                data_path=synthetic_data_dir,
                feature_a='f0',
                feature_b='label',
            ),
        )
        assert signal['score'] > noise['score'], \
            f"Signal feature f30 ({signal['score']:.4f}) should outscore random f0 ({noise['score']:.4f})"


class TestInteractionInfo:
    def test_xor_synergy(self, xor_data_dir):
        """XOR-like Y=(X1+X2)%3 should show negative II (synergy)."""
        from outrank.mcp_server import outrank_compute_interaction_info

        result = json.loads(
            outrank_compute_interaction_info(
                data_path=xor_data_dir,
                feature_a='x1',
                feature_b='x2',
                label_column='label',
            ),
        )
        assert result['interaction_info'] < 0, \
            f"XOR synergy should yield negative II, got {result['interaction_info']:.4f}"
        assert 'synergy' in result['interpretation']


class TestRankFeatures:
    def test_end_to_end_with_jmi_and_summarize(self, synthetic_data_dir, output_dir):
        """End-to-end: rank with JMI → verify outputs → summarize.

        Runs as a single test because pathos ProcessingPool has global state
        and cannot be reliably re-created within the same process. Calling
        outrank_rank_features multiple times in one pytest process triggers
        'Pool not running' errors. Consolidating into one call avoids this.
        """
        from outrank.mcp_server import outrank_rank_features, outrank_summarize_results

        result = json.loads(
            outrank_rank_features(
                data_path=synthetic_data_dir,
                label_column='label',
                target_ranking_only=True,
                compute_jmi=True,
                jmi_top_k=10,
                subsampling=1,
                num_threads=2,
                output_folder=output_dir,
            ),
        )

        # Verify pairwise ranking results
        assert 'pairwise_ranks' in result
        assert len(result['pairwise_ranks']) > 0
        assert result['elapsed_seconds'] > 0
        assert os.path.exists(os.path.join(output_dir, 'pairwise_ranks.tsv'))

        # Verify JMI results
        assert 'jmi_ranking' in result, 'JMI ranking should be present in results'
        assert len(result['jmi_ranking']) > 0

        # Verify summarize_results on the same output
        summary = json.loads(
            outrank_summarize_results(
                output_folder=output_dir,
                top_n=5,
            ),
        )
        assert 'top_pairwise_ranks' in summary
        assert len(summary['top_pairwise_ranks']) == 5
        assert 'markdown_table' in summary
        assert '|' in summary['markdown_table']
