#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$SCRIPT_DIR/test_data_synthetic"
OUTPUT_DIR="/tmp/outrank_jmi_demo"
rm -rf "$OUTPUT_DIR"

if [ ! -f "$DATA_DIR/data.csv" ]; then
  echo "Test data not found. Generating..."
  python -m outrank --task data_generator --num_synthetic_rows 100000
  DATA_DIR="test_data_synthetic"
fi

python -m outrank --task ranking \
  --data_path "$DATA_DIR" \
  --data_source csv-raw \
  --heuristic MI-numba-randomized \
  --compute_jmi True \
  --compute_interaction_info True \
  --output_folder "$OUTPUT_DIR"

echo ""
echo "=== JMI Feature Ranking (top 10) ==="
head -11 "$OUTPUT_DIR/jmi_feature_ranking.tsv"

echo ""
echo "=== Most Synergistic Pairs (II < 0) ==="
tail -n +2 "$OUTPUT_DIR/interaction_information.tsv" | sort -t$'\t' -k3 -g | head -5

echo ""
echo "=== Most Redundant Pairs (II > 0) ==="
tail -n +2 "$OUTPUT_DIR/interaction_information.tsv" | sort -t$'\t' -k3 -gr | head -5

echo ""
echo "Output files in $OUTPUT_DIR/"
ls -lh "$OUTPUT_DIR"/*.tsv
