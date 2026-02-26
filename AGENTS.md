# AGENTS.md

This file documents the OutRank repository structure, conventions, and guidelines for AI-assisted development (Claude Code, Copilot, or any other agentic coding tool).

## Repository overview

OutRank is a feature ranking system for large, sparse, categorical datasets. It implements cardinality-aware mutual information variants, optimized with Numba JIT compilation, and streams data in minibatches to handle datasets that don't fit in memory.

- **Language**: Python 3.9+
- **Build**: `setup.py` (setuptools), entry point `outrank` -> `outrank/__main__.py:main`
- **Version**: defined in `setup.py` (`version='0.97.7'`)
- **License**: BSD

## Directory structure

```
outrank/
├── outrank/                        # Main package
│   ├── __main__.py                 # CLI entry point, argparse, task routing
│   ├── core_ranking.py             # Batch orchestration, feature interaction scoring
│   ├── core_utils.py               # Data parsing, schema types, utilities
│   ├── core_selftest.py            # Self-test data/assertions
│   ├── task_ranking.py             # Ranking workflow (process pool, checkpointing)
│   ├── task_summary.py             # Post-processing: normalize, aggregate, filter
│   ├── task_generators.py          # Synthetic dataset generation entry point
│   ├── task_instance_ranking.py    # Per-row quality scoring
│   ├── task_visualization.py       # Visualization wrapper
│   ├── task_selftest.py            # End-to-end self-validation
│   ├── algorithms/
│   │   ├── importance_estimator.py # Heuristic router (10+ scoring algorithms)
│   │   ├── feature_ranking/
│   │   │   ├── ranking_mi_numba.py         # Numba MI (base implementation)
│   │   │   ├── ranking_mi_numba_opt.py     # Numba MI (grouped, cache-friendly)
│   │   │   ├── ranking_mi_numba_cmi.py     # Conditional MI, interaction info, JMI primitives
│   │   │   ├── ranking_mi_multivalue.py    # MI for multivalue/set features
│   │   │   └── ranking_cov_alignment.py    # Coverage-based heuristic
│   │   ├── sketches/
│   │   │   ├── counting_ultiloglog.py      # HyperLogLog cardinality estimator
│   │   │   ├── counting_counters_ordinary.py # Bounded frequency counter
│   │   │   └── counting_cms.py             # Count-Min Sketch
│   │   └── synthetic_data_generators/
│   │       ├── generator_naive.py          # Simple random matrix generator
│   │       └── cc_generator.py             # Controllable categorical generator
│   ├── feature_transformations/
│   │   ├── ranking_transformers.py         # Transformer pipeline (noise, generic)
│   │   └── feature_transformer_vault/
│   │       ├── default_transformers.py     # Preset transformer specs
│   │       └── fw_transformers.py          # Field-weighted transformers
│   └── visualizations/
│       └── ranking_visualization.py        # Dendrograms, t-SNE, heatmaps
├── tests/                          # Unit and integration tests
│   ├── mi_numba_test.py            # Numba MI correctness
│   ├── mi_numba_opt_test.py        # Optimized MI correctness
│   ├── mi_numba_cmi_test.py        # Conditional MI, interaction info, JMI
│   ├── multivalue_mi_test.py       # Multivalue feature MI
│   ├── hll_test.py                 # HyperLogLog accuracy
│   ├── cms_test.py                 # Count-Min Sketch
│   ├── cov_heu_test.py             # Coverage heuristic
│   ├── cc_generator_test.py        # Synthetic data generator
│   ├── json_transformers_test.py   # Custom transformer JSON parsing
│   ├── ranking_module_test.py      # Core ranking module
│   ├── integration_tests.py        # Cross-component integration
│   └── data_io_test.py             # Data parsing/IO
├── benchmarks/                     # Performance benchmarks
│   ├── analyse_rankings.py
│   ├── generator_naive.py
│   ├── generator_second_order.py
│   └── generator_third_order.py
├── scripts/
│   ├── run_unit_tests.sh           # pytest runner
│   ├── run_benchmarks.sh
│   └── run_minimal.sh
├── examples/                       # Usage examples
├── docs/                           # Documentation source
├── .github/workflows/              # CI pipelines
│   ├── python-package.yml          # Build + lint + test (Py 3.9/3.10/3.11)
│   ├── python-unit.yml             # Unit tests only
│   ├── selftest.yml                # End-to-end self-test
│   └── benchmarks.yml              # Performance regression
├── setup.py                        # Package metadata + dependencies
├── pyproject.toml                  # autopep8 config
├── requirements.txt                # Runtime dependencies
└── README.md                       # Project overview + citation
```

## Data flow

```
Input (CSV/TSV/VW)
  → task_ranking.outrank_task_conduct_ranking()
    → core_utils.get_dataset_info()           # parse schema
    → core_ranking.estimate_importances_minibatches()  # stream batches
      → core_ranking.compute_batch_ranking()  # per-batch:
        ├─ ranking_transformers (feature engineering)
        ├─ counting_ultiloglog (cardinality estimation via HLL)
        └─ core_ranking.mixed_rank_graph()
           └─ ProcessingPool.amap(importance_estimator.get_importances_estimate_pairwise)
              └─ importance_estimator.conduct_feature_ranking()
                 └─ [selected heuristic: MI-numba-opt / multivalue / AMI / ...]
  → task_summary.outrank_task_result_summary()   # normalize + aggregate
  → task_visualization.outrank_task_visualize_results()  # plots
Output: pairwise_ranks.tsv, feature_singles.tsv, visualizations
```

## Parallelism model

- **Process pool** via `pathos.multiprocessing.ProcessingPool`
- One worker per feature-pair combination (not per batch)
- Global state (cardinality, counts) is per-process (process isolation), aggregated post-batch
- Main thread polls with `time.sleep(4)` between checks

## Conventions and rules for agents

### Code style
- **autopep8** is configured in `pyproject.toml` (in-place, ignore W690)
- **flake8** is enforced in CI: syntax errors are blocking, style warnings are advisory (max-line-length=127, max-complexity=10)
- **pre-commit** is a listed dependency; respect any hooks present
- Imports: `from __future__ import annotations` is used throughout

### Testing
- Framework: **pytest**
- Run: `python -m pytest ./tests/*.py` or `bash scripts/run_unit_tests.sh`
- CI matrix: Python 3.9, 3.10, 3.11 on ubuntu-latest
- Self-test: `outrank --task selftest` (generates synthetic data, runs ranking, validates output)
- All new code touching algorithms **must** have a corresponding test in `tests/`

### Performance-critical code
- Numba `@njit` decorated functions in `ranking_mi_numba.py`, `ranking_mi_numba_opt.py`, and `ranking_mi_numba_cmi.py` — do not introduce Python objects, dynamic typing, or unsupported NumPy operations inside `@njit` functions
- The `ranking_mi_numba_opt` variant uses pre-grouped indices for cache locality — maintain this property when modifying
- `ranking_mi_numba_cmi.py` has two CMI paths: (1) **contingency table** (fast, O(n + dx*dy*dz), used when `cardinality_correction=False` and product <= 2M), (2) **per-Z-group** (fallback, for cardinality correction or extreme cardinality). Do not remove the fallback — it handles edge cases
- Numba `@njit(cache=True)` functions **cannot be reliably imported cross-module** — `ranking_mi_numba_cmi.py` copies `_build_groups`/`_compute_entropies_grouped` from `ranking_mi_numba_opt.py`. Keep them in sync
- JMI greedy selection uses **incremental score accumulation** — each step only computes CMI for the newly-selected feature, not all selected features. Do not regress to naive O(k^3) loop
- HyperLogLog in `counting_ultiloglog.py` uses a warm-up/exact-count hybrid — the switchover threshold is `m/2`
- `PrimitiveConstrainedCounter` has a hard size bound (default 30k) — this is intentional backpressure, not a bug

### Key dependencies
| Package | Purpose |
|---------|---------|
| numba | JIT compilation for MI scoring hot paths |
| pathos | Process pool for parallel feature scoring |
| xxhash | Fast hashing for feature combination construction |
| pandas/numpy | Data manipulation |
| scipy/sklearn | Statistical functions, surrogate models |
| zstandard | Compressed data support |
| matplotlib/seaborn | Visualization |

### What NOT to do
- Do not add type stubs, docstrings, or comments to code you did not modify
- Do not refactor the global state pattern in `core_ranking.py` — process isolation is load-bearing
- Do not replace `pathos` with `multiprocessing` — `pathos` is required for pickling closures across process boundaries
- Do not add `@jit` (as opposed to `@njit`) — object mode defeats the purpose
- Do not introduce new dependencies without explicit approval
- Do not modify CI workflow files without explicit approval
- Do not commit generated outputs (`ranking_outputs/`, `test_data_synthetic/`, `*.pdf`, `*.html`, `*.json` in examples/)

### Commit and PR conventions
- Commit messages: imperative mood, concise, focused on *why*
- PRs target `main` branch
- CI must pass (lint + unit tests + selftest) before merge
- Branch naming: descriptive slugs (e.g., `fix-multivalue-reshape`, `add-pearson-heuristic`)

## Heuristic registry

These are the scoring algorithms routed through `importance_estimator.conduct_feature_ranking()`:

| Heuristic ID | Module | Notes |
|-------------|--------|-------|
| `MI-numba-randomized` | `ranking_mi_numba` | Base Numba MI with stratified subsampling |
| `MI-numba-randomized-opt` | `ranking_mi_numba_opt` | Grouped variant, better cache behavior |
| `MI` | sklearn `mutual_info_classif` | Exact MI, slower |
| `AMI` | sklearn `adjusted_mutual_info_score` | Adjusted for chance |
| `MI-multivalue-set_based` | `ranking_mi_multivalue` | Set-theoretic MI for multivalue features |
| `MI-multivalue-jaccard` | `ranking_mi_multivalue` | Jaccard similarity MI |
| `MI-multivalue-overlap` | `ranking_mi_multivalue` | Overlap coefficient MI |
| `surrogate-SGD` | importance_estimator | OneHot + LogisticRegression proxy |
| `surrogate-SVM` | importance_estimator | LinearSVC proxy |
| `surrogate-SGD-RP` | importance_estimator | Random projection + SGD |
| `correlation-Pearson` | importance_estimator | Pearson correlation |
| `max-value-coverage` | `ranking_cov_alignment` | Most-frequent-pair proportion |
| `Constant` | importance_estimator | Returns 0 (placeholder) |
| `3MR` | importance_estimator | mRMR-style multi-objective (relevance - redundancy + relational) |
| `CMI` (via `--compute_jmi`) | `ranking_mi_numba_cmi` | Conditional MI I(X;Y\|Z), JMI greedy selection. Contingency table fast path (~2x vs per-group) |
| `II` (via `--compute_interaction_info`) | `ranking_mi_numba_cmi` | Interaction information II(X_i,X_j;Y): synergy vs redundancy (Jakulin & Bratko convention) |

## Pre-commit / CI test hook

Any code addition or modification to `outrank/` or `tests/` **must** pass the following before merge:

```bash
# 1. Unit tests (all 182+ tests)
python -m pytest tests/ -v

# 2. Lint (blocking: syntax errors and undefined names)
flake8 outrank/ --count --select=E9,F63,F7,F82 --show-source --statistics

# 3. Lint (advisory: style warnings, non-blocking)
flake8 outrank/ --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics

# 4. Selftest (end-to-end: data generation -> ranking -> output validation)
python -m outrank --task data_generator --num_synthetic_rows 100000
python -m outrank --task ranking --data_path test_data_synthetic --data_source csv-raw --heuristic MI-numba-randomized --output_folder ranking_outputs
python -c "
import pandas as pd
dfx = pd.read_csv('ranking_outputs/pairwise_ranks.tsv', sep='\t')
assert dfx.shape[0] == 201 and dfx.shape[1] == 3
print('Selftest OK')
"
rm -rf ranking_outputs test_data_synthetic
```

Agents **must** run steps 1-2 after every code change. Step 4 should be run before creating a PR.

## Agentic activity log

All significant changes made by AI agents (Claude Code, Copilot, etc.) should be documented below with date, branch, and summary. This provides an audit trail for AI-assisted modifications.

| Date | Branch | Agent | Summary |
|------|--------|-------|---------|
| 2025-* | `copilot/fix-mildly-annoying-logs` | Copilot | PR #112 — fix log noise |
| 2025-* | `multivalue-improvements` | Claude Code | Fix multivalue MI reshape handling, AMI/Pearson vector reshaping, optimize redundant reshapes |
| 2025-* | `copilot/fix-*` | Copilot | PR #108 — bug fix |
| 2025-* | `add-llm-optimized-MI` | External contributor | PR #110 — LLM-optimized MI variant |
| 2026-02-26 | `multivalue-improvements` | Claude Code | Created AGENTS.md documenting repository structure and agentic guidelines |
| 2026-02-26 | `multivalue-improvements` | Claude Code | Added Conditional MI, Interaction Information, JMI feature selection (ranking_mi_numba_cmi.py, nonmyopic stub, CLI flags) |
| 2026-02-26 | `multivalue-improvements` | Claude Code | Optimized CMI: incremental JMI scores (O(k^3)->O(k^2)), contingency table fast path (~2x), label exclusion fix. JMI overhead: <1% |
