# Feature Evolution via Ranking

This script facilitates the process of feature evolution through iterative ranking using the `outrank` tool. It automates the process of running multiple iterations of feature ranking, extracting the best features, and updating the model specifications accordingly.

## Overview

The script performs the following steps:
1. **Initialization**: Sets up the initial model specification directory and creates the initial model JSON file.
2. **Iteration**: Runs the `outrank` task for a specified number of iterations.
3. **Feature Extraction**: Processes the results of each iteration to extract the best feature.
4. **Model Update**: Updates the model specification JSON with the newly identified best feature.

## Prerequisites

- Ensure that the `outrank` tool is installed and accessible from the command line.
- Python 3.6 or higher.
- Required Python packages: `pandas`, `argparse`, `json`, `shutil`, and `logging`.

## Installation

Install the required Python packages using pip (`pip install outrank --upgrade`)

---

# JSON-Based Feature Transformers

This directory also contains example JSON files for specifying custom feature transformations in OutRank.

## JSON Transformer Overview

OutRank now supports loading feature transformers from JSON specification files in addition to the built-in presets. This allows users to define custom numpy-based transformations without modifying the source code.

## JSON Format

The JSON format is simple: a dictionary where keys are transformer names and values are numpy expressions:

```json
{
    "_tr_sqrt": "np.sqrt(X)",
    "_tr_log": "np.log(X + 1)",
    "_tr_custom": "np.tanh(X)"
}
```

## JSON Transformer Examples

### `simple_transformers.json`
Basic mathematical transformations including square root, logarithm, square, absolute value, and exponential.

### `custom_transformers.json`
Advanced transformations including sigmoid, tanh, ReLU, normalization, z-score standardization, and other custom functions.

## Usage

### Command Line Interface

```bash
# Use JSON transformers only
outrank --transformers examples/simple_transformers.json --data_path mydata/ --data_source csv-raw

# Combine preset with JSON transformers
outrank --transformers default,examples/custom_transformers.json --data_path mydata/ --data_source csv-raw
```

### Python API

```python
from outrank.feature_transformations.ranking_transformers import FeatureTransformerGeneric

# JSON transformers only
transformer = FeatureTransformerGeneric(
    numeric_columns={'feature1', 'feature2'}, 
    preset='examples/simple_transformers.json'
)

# Combine with presets
transformer = FeatureTransformerGeneric(
    numeric_columns={'feature1', 'feature2'}, 
    preset='minimal,examples/custom_transformers.json'
)
```

## Creating Custom Transformers

1. Create a JSON file with your transformer specifications
2. Use valid numpy expressions where `X` represents the input feature array
3. Follow the naming convention `_tr_*` for transformer names
4. Ensure all expressions are strings in the JSON

### Example Custom Transformer

```json
{
    "_tr_my_custom": "np.log(np.abs(X) + 1) * np.sqrt(X)",
    "_tr_sigmoid_scaled": "1 / (1 + np.exp(-X * 0.1))",
    "_tr_percentile_rank": "np.searchsorted(np.sort(X), X) / len(X)"
}
```
