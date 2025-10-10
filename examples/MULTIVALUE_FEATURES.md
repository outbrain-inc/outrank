# Multivalue Features in OutRank

This directory contains examples for working with multivalue features in OutRank.

## Important: Delimiter Choice

**Use `_` (underscore) as delimiter for multivalue features, NOT `,` (comma)!**

### Why underscore instead of comma?

Since OutRank works with CSV files, using comma as a delimiter would cause parsing conflicts:

```csv
# BAD: Ambiguous - is this 3 columns or 2?
user_id,interests,skills
1,sports,music,python,sql

# GOOD: Clear structure with underscore delimiter
user_id,interests,skills
1,sports_music,python_sql
```

## Files

- **`multivalue_data.csv`** - Sample CSV data with multivalue features using `_` delimiter
- **`multivalue_end_to_end.py`** - Complete end-to-end Python API example with 5 demonstrations
- **`multivalue_example.py`** - Basic Python usage examples
- **`run_multivalue_example.sh`** - Shell script demonstrating OutRank CLI with multivalue heuristics

## Quick Start

### Using OutRank CLI (Recommended)

```bash
# Run OutRank with multivalue MI heuristics
./run_multivalue_example.sh

# This will execute:
# outrank --task all --data_path examples/multivalue_data.csv \
#         --data_source csv-raw --heuristic MI-multivalue-set \
#         --target_ranking_only True --num_threads 8 \
#         --output_folder ./ranking_outputs_multivalue
```

### Using Python API

```bash
# Run the Python end-to-end example
python3 multivalue_end_to_end.py
```

## Usage Examples

### 1. Basic Multivalue MI Computation

```python
import numpy as np
from outrank.algorithms.feature_ranking.ranking_mi_multivalue import multivalue_mutual_info_estimator

# Data with underscore delimiter
interests = np.array(['sports_music', 'music_tech', 'sports_tech'])
skills = np.array(['python_sql', 'java_sql', 'python_java'])

# Compute MI with different algorithms
score = multivalue_mutual_info_estimator(interests, skills, algorithm='set_based', delimiter='_')
print(f"MI score: {score}")
```

### 2. Integration with OutRank's API

```python
from outrank.algorithms.importance_estimator import conduct_feature_ranking

class Args:
    def __init__(self):
        self.heuristic = 'MI-multivalue-set'  # or 'MI-multivalue-jaccard', 'MI-multivalue-overlap'
        self.mi_stratified_sampling_ratio = 1.0

args = Args()
score = conduct_feature_ranking(interests, target, args)
```

### 3. CSV Data Format

Your CSV should look like this:

```csv
user_id,interests,skills,purchased,satisfaction
1,sports_music,python_sql,laptop_phone,high
2,music_tech,java_sql,phone_tablet,high
3,sports_tech,python_java,laptop_tablet,medium
```

Each multivalue field uses `_` to separate values within that field.

## Available Algorithms

Three algorithms are available for multivalue MI:

1. **`set_based`** (recommended) - Direct set-based mutual information computation
2. **`jaccard`** - Analyzes structural patterns via neighbor row analysis
3. **`overlap`** - Based on set overlap and cross-row co-occurrence patterns

## Heuristics for OutRank

When using OutRank's CLI or main API, use these heuristic names:

- `MI-multivalue-set` (recommended)
- `MI-multivalue-jaccard`
- `MI-multivalue-overlap`

### Example CLI Usage

```bash
outrank \
    --task all \
    --data_path examples/multivalue_data.csv \
    --data_source csv-raw \
    --heuristic MI-multivalue-set \
    --target_ranking_only True \
    --num_threads 8 \
    --output_folder ./ranking_outputs_multivalue \
    --subsampling 100
```

## Key Features

✓ **No expansion needed** - Works directly on multivalue features  
✓ **Preserves co-occurrence** - Maintains information about values appearing together  
✓ **Handles sequential patterns** - Detects information even with empty row-wise intersections  
✓ **CSV-safe delimiter** - Using `_` avoids parsing conflicts  
✓ **Backward compatible** - All existing OutRank functionality preserved  

## Examples Included

The `multivalue_end_to_end.py` script includes 5 comprehensive examples:

1. **Basic Multivalue MI Computation** - Loading and processing multivalue data
2. **Integration with OutRank's API** - Using with `conduct_feature_ranking`
3. **Sequential Patterns** - Handling data without direct intersections
4. **Delimiter Importance** - Why `_` is better than `,`
5. **Comparison with Expansion** - Multivalue MI vs one-hot encoding

## Testing

Run the test suite:

```bash
python -m unittest tests.multivalue_mi_test -v
```

All 16 tests should pass, including tests for:
- Basic functionality
- Sequential pattern detection
- Custom delimiter support
- Edge cases (empty inputs, mismatched lengths, etc.)

## Performance Notes

- Direct multivalue MI is more efficient than one-hot expansion
- No need to create sparse, high-dimensional feature matrices
- Preserves semantic relationships between multivalue features

## Further Reading

See the main OutRank documentation for more information about:
- Feature ranking algorithms
- Integration with the OutRank CLI
- Advanced configuration options
