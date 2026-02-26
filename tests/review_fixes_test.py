"""Tests for the review-fix changes.

Covers: MI clamping (Fix 3), label validation (Fix 2), JMI top_k clamping
(Fix 6), MinibatchResult namedtuple (Fix 4), MCP guards (Fix 5), CLI
defaults unification (Fix 7).
"""
from __future__ import annotations

import argparse
import os
import shutil
import tempfile

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Fix 3: MI clamping under cardinality correction
# ---------------------------------------------------------------------------

class TestMIClampNegativeCC:
    """Cardinality correction can produce slightly negative MI on independent
    high-cardinality features. Verify the clamp to >= 0."""

    def test_mi_clamp_negative_cc(self):
        from outrank.algorithms.feature_ranking.ranking_mi_numba_opt import (
            mutual_info_estimator_numba_opt,
        )

        rng = np.random.RandomState(42)
        n = 500
        # Independent high-cardinality features — MI should be ~0
        X = rng.randint(0, 200, size=n).astype(np.int32)
        Y = rng.randint(0, 200, size=n).astype(np.int32)

        mi = float(mutual_info_estimator_numba_opt(Y, X, np.float32(1.0), True))
        assert mi >= 0.0, f'MI with CC should be >= 0, got {mi}'

    def test_mi_clamp_negative_cc_contingency(self):
        """Same test but targeting the contingency table CC path (dx*dy <= 2M)."""
        from outrank.algorithms.feature_ranking.ranking_mi_numba_opt import (
            _compute_mi_contingency_cc,
        )

        rng = np.random.RandomState(99)
        n = 300
        X = rng.randint(0, 50, size=n).astype(np.int32)
        Y = rng.randint(0, 50, size=n).astype(np.int32)

        mi = float(_compute_mi_contingency_cc(Y, X, np.int32(n), np.int32(50), np.int32(50)))
        assert mi >= 0.0, f'MI (contingency CC) should be >= 0, got {mi}'

    def test_cmi_clamp_negative_cc(self):
        """CMI path in ranking_mi_numba_cmi.py — CC via _compute_entropies_grouped."""
        from outrank.algorithms.feature_ranking.ranking_mi_numba_cmi import (
            _mi_from_arrays,
        )

        rng = np.random.RandomState(7)
        n = 500
        X = rng.randint(0, 200, size=n).astype(np.int32)
        Y = rng.randint(0, 200, size=n).astype(np.int32)

        mi = float(_mi_from_arrays(Y, X, np.float32(1.0), True))
        assert mi >= 0.0, f'MI (CMI module, CC) should be >= 0, got {mi}'

    def test_cmi_clamp_negative_cc_contingency_local(self):
        """CMI module contingency CC path."""
        from outrank.algorithms.feature_ranking.ranking_mi_numba_cmi import (
            _compute_mi_contingency_cc_local,
        )

        rng = np.random.RandomState(11)
        n = 300
        X = rng.randint(0, 50, size=n).astype(np.int32)
        Y = rng.randint(0, 50, size=n).astype(np.int32)

        mi = float(_compute_mi_contingency_cc_local(Y, X, np.int32(n), np.int32(50), np.int32(50)))
        assert mi >= 0.0, f'MI (CMI contingency CC) should be >= 0, got {mi}'

    def test_mi_positive_signal_unchanged(self):
        """Ensure clamping doesn't break genuine positive MI."""
        from outrank.algorithms.feature_ranking.ranking_mi_numba_opt import (
            mutual_info_estimator_numba_opt,
        )

        rng = np.random.RandomState(42)
        n = 5000
        X = rng.randint(0, 5, size=n).astype(np.int32)
        Y = X.copy()
        # Flip 10% to add noise
        flip = rng.random(n) < 0.1
        Y[flip] = rng.randint(0, 5, size=flip.sum()).astype(np.int32)

        mi = float(mutual_info_estimator_numba_opt(Y, X, np.float32(1.0), True))
        assert mi > 0.1, f'Correlated features should have positive MI, got {mi}'


# ---------------------------------------------------------------------------
# Fix 2: Label column validation
# ---------------------------------------------------------------------------

class TestLabelColumnValidation:
    def test_missing_label_column_raises(self):
        """task_ranking should raise ValueError when label column is not in dataset."""
        from outrank.task_ranking import outrank_task_conduct_ranking

        tmpdir = tempfile.mkdtemp(prefix='outrank_label_test_')
        try:
            csv_dir = os.path.join(tmpdir, 'data_dir')
            os.makedirs(csv_dir)
            df = pd.DataFrame({'feat1': [1, 2, 3], 'feat2': [4, 5, 6], 'target': [0, 1, 0]})
            df.to_csv(os.path.join(csv_dir, 'data.csv'), index=False)

            args = argparse.Namespace(
                task='ranking',
                data_path=csv_dir,
                data_source='csv-raw',
                label_column='nonexistent_label',
                heuristic='MI-numba-randomized',
                output_folder=os.path.join(tmpdir, 'out'),
                disable_tqdm='True',
                num_threads=1,
                subsampling=1,
                minibatch_size=2**14,
                combination_number_upper_bound=2**15,
                missing_value_symbols=',{}',
                include_noise_baseline_features='False',
                include_cardinality_in_feature_names='False',
                target_ranking_only='True',
                transformers='none',
                feature_set_focus=None,
                interaction_order=1,
                reference_model_JSON='',
                explode_multivalue_features='False',
                subfeature_mapping='False',
                max_unique_hist_constraint=30000,
                mi_stratified_sampling_ratio=1.0,
                compute_jmi='False',
                jmi_top_k=50,
                compute_interaction_info='False',
                interaction_info_top_k=30,
                rare_value_count_upper_bound=1,
                tldr='False',
            )

            with pytest.raises(ValueError, match='nonexistent_label'):
                outrank_task_conduct_ranking(args)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Fix 6: JMI top_k clamping
# ---------------------------------------------------------------------------

class TestJMITopKClamping:
    def test_jmi_top_k_exceeds_features(self):
        """JMI should not crash when top_k > number of features."""
        from outrank.algorithms.importance_estimator import get_importances_estimate_nonmyopic

        rng = np.random.RandomState(42)
        n = 500
        # Only 3 features, but top_k=50
        df = pd.DataFrame({
            'f0': rng.randint(0, 5, size=n),
            'f1': rng.randint(0, 5, size=n),
            'f2': rng.randint(0, 5, size=n),
            'label': rng.randint(0, 2, size=n),
        })
        # Integer-encode via factorize
        for col in df.columns:
            df[col] = pd.factorize(df[col])[0].astype(np.int32)

        pairwise = {}
        for f in ['f0', 'f1', 'f2']:
            pairwise[(f, 'label')] = rng.random()
            pairwise[('label', f)] = pairwise[(f, 'label')]
        # label diagonal
        pairwise[('label', 'label')] = 1.0

        args = argparse.Namespace(
            label_column='label',
            heuristic='MI-numba-randomized',
            mi_stratified_sampling_ratio=1.0,
        )

        result = get_importances_estimate_nonmyopic(
            args, df, pairwise_mi_dict=pairwise, top_k=50,
        )
        assert result is not None
        assert len(result) <= 3, f'Should select at most 3 features, got {len(result)}'

    def test_ii_top_k_exceeds_features(self):
        """compute_interaction_information_for_pairs with top_k > n_features."""
        from outrank.algorithms.importance_estimator import compute_interaction_information_for_pairs

        rng = np.random.RandomState(42)
        n = 500
        df = pd.DataFrame({
            'f0': rng.randint(0, 5, size=n),
            'f1': rng.randint(0, 5, size=n),
            'label': rng.randint(0, 2, size=n),
        })
        for col in df.columns:
            df[col] = pd.factorize(df[col])[0].astype(np.int32)

        pairwise = {
            ('f0', 'label'): 0.5, ('label', 'f0'): 0.5,
            ('f1', 'label'): 0.3, ('label', 'f1'): 0.3,
            ('label', 'label'): 1.0,
        }

        args = argparse.Namespace(
            label_column='label',
            heuristic='MI-numba-randomized',
            mi_stratified_sampling_ratio=1.0,
        )

        result = compute_interaction_information_for_pairs(
            df, args, pairwise_mi_dict=pairwise, top_k=100,
        )
        assert result is not None
        assert len(result) == 1  # Only 1 pair possible with 2 features


# ---------------------------------------------------------------------------
# Fix 4: MinibatchResult namedtuple
# ---------------------------------------------------------------------------

class TestMinibatchResult:
    def test_fields_exist(self):
        from outrank.core_utils import MinibatchResult

        r = MinibatchResult(
            step_timing_checkpoints=[{'t': 1.0}],
            mutual_information_estimates=None,
            cardinality_object={},
            bounds_object_storage=[],
            memory_object_storage=[],
            coverage_object={},
            rare_value_storage={},
            prior_comb_counts={},
            item_counts={},
            jmi_ranking=None,
            interaction_info=None,
        )
        assert r.step_timing_checkpoints == [{'t': 1.0}]
        assert r.mutual_information_estimates is None
        assert r.jmi_ranking is None
        assert r.interaction_info is None

    def test_optional_fields_default_none(self):
        from outrank.core_utils import MinibatchResult

        r = MinibatchResult(
            step_timing_checkpoints=[],
            mutual_information_estimates=None,
            cardinality_object={},
            bounds_object_storage=[],
            memory_object_storage=[],
            coverage_object={},
            rare_value_storage={},
            prior_comb_counts={},
            item_counts={},
        )
        assert r.jmi_ranking is None
        assert r.interaction_info is None


# ---------------------------------------------------------------------------
# Fix 5: MCP server request guards
# ---------------------------------------------------------------------------

class TestMCPGuards:
    def test_invalid_data_path(self):
        from outrank.mcp_adapters import build_ranking_namespace, run_ranking_safe

        args = build_ranking_namespace(data_path='/nonexistent/path/xyz')
        result = run_ranking_safe(args)
        assert 'error' in result
        assert 'does not exist' in result['error']

    def test_none_data_path(self):
        from outrank.mcp_adapters import build_ranking_namespace, run_ranking_safe

        args = build_ranking_namespace(data_path=None)
        result = run_ranking_safe(args)
        assert 'error' in result
        assert 'must not be None' in result['error']

    def test_file_size_guard(self):
        from outrank.mcp_adapters import validate_data_path, MAX_FILE_SIZE_MB

        tmpdir = tempfile.mkdtemp(prefix='outrank_size_test_')
        try:
            # Create a mock "large" file by patching the threshold
            import outrank.mcp_adapters as adapters
            original = adapters.MAX_FILE_SIZE_MB
            adapters.MAX_FILE_SIZE_MB = 0  # 0 MB limit -> any file is "too large"

            csv_path = os.path.join(tmpdir, 'big.csv')
            with open(csv_path, 'w') as f:
                f.write('a,b\n1,2\n')

            with pytest.raises(ValueError, match='exceeding'):
                validate_data_path(csv_path)

            adapters.MAX_FILE_SIZE_MB = original
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_valid_data_path_passes(self):
        from outrank.mcp_adapters import validate_data_path

        tmpdir = tempfile.mkdtemp(prefix='outrank_valid_test_')
        try:
            csv_dir = os.path.join(tmpdir, 'data')
            os.makedirs(csv_dir)
            csv_path = os.path.join(csv_dir, 'data.csv')
            with open(csv_path, 'w') as f:
                f.write('a,b,label\n1,2,0\n3,4,1\n')

            # Should not raise
            validate_data_path(csv_dir)
            validate_data_path(csv_path)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Fix 7: CLI defaults unification
# ---------------------------------------------------------------------------

class TestCLIDefaultsUnification:
    def test_cli_defaults_importable(self):
        from outrank.cli_defaults import CLI_DEFAULTS

        assert isinstance(CLI_DEFAULTS, dict)
        assert 'label_column' in CLI_DEFAULTS
        assert CLI_DEFAULTS['label_column'] == 'label'
        assert CLI_DEFAULTS['heuristic'] == 'MI-numba-randomized'

    def test_mcp_inherits_cli_defaults(self):
        """MCP adapter should inherit values from CLI_DEFAULTS for shared keys."""
        from outrank.cli_defaults import CLI_DEFAULTS
        from outrank.mcp_adapters import build_ranking_namespace

        ns = build_ranking_namespace()
        # These should match CLI_DEFAULTS (not diverge)
        assert ns.label_column == CLI_DEFAULTS['label_column']
        assert ns.heuristic == CLI_DEFAULTS['heuristic']
        assert ns.minibatch_size == CLI_DEFAULTS['minibatch_size']
        assert ns.combination_number_upper_bound == CLI_DEFAULTS['combination_number_upper_bound']
        assert ns.missing_value_symbols == CLI_DEFAULTS['missing_value_symbols']
        assert ns.compute_jmi == CLI_DEFAULTS['compute_jmi']
        assert ns.jmi_top_k == CLI_DEFAULTS['jmi_top_k']

    def test_cli_argparse_uses_cli_defaults(self):
        """CLI argparse defaults should come from CLI_DEFAULTS, not hardcoded."""
        from outrank.cli_defaults import CLI_DEFAULTS
        import outrank.__main__ as main_mod

        # Verify the module-level D alias points to CLI_DEFAULTS
        assert main_mod.D is CLI_DEFAULTS

    def test_cli_mcp_parity_on_shared_keys(self):
        """All shared keys between CLI and MCP must have the same base default."""
        from outrank.cli_defaults import CLI_DEFAULTS
        from outrank.mcp_adapters import build_ranking_namespace

        ns = build_ranking_namespace()
        # MCP overrides a few keys for headless mode; all others must match
        mcp_overrides = {
            'task', 'output_folder', 'data_source', 'num_threads',
            'disable_tqdm', 'tldr', 'num_synthetic_rows',
        }
        for key, cli_val in CLI_DEFAULTS.items():
            if key in mcp_overrides:
                continue
            mcp_val = getattr(ns, key)
            assert mcp_val == cli_val, (
                f"Default divergence for '{key}': CLI={cli_val!r}, MCP={mcp_val!r}"
            )


# ---------------------------------------------------------------------------
# Fix 5 extended: MCP tools that directly read CSV also validate
# ---------------------------------------------------------------------------

class TestMCPToolValidation:
    """Verify that MCP tools that directly read CSV files
    call validate_data_path before processing."""

    def test_dataset_info_rejects_nonexistent(self):
        pytest.importorskip('mcp', reason='mcp package not installed')
        from outrank.mcp_server import outrank_dataset_info

        with pytest.raises(ValueError, match='does not exist'):
            outrank_dataset_info(data_path='/nonexistent/path/xyz')

    def test_score_pair_rejects_nonexistent(self):
        pytest.importorskip('mcp', reason='mcp package not installed')
        from outrank.mcp_server import outrank_score_pair

        with pytest.raises(ValueError, match='does not exist'):
            outrank_score_pair(
                data_path='/nonexistent/path/xyz',
                feature_a='a', feature_b='b',
            )

    def test_interaction_info_rejects_nonexistent(self):
        pytest.importorskip('mcp', reason='mcp package not installed')
        from outrank.mcp_server import outrank_compute_interaction_info

        with pytest.raises(ValueError, match='does not exist'):
            outrank_compute_interaction_info(
                data_path='/nonexistent/path/xyz',
                feature_a='a', feature_b='b',
            )
