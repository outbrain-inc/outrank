#!/bin/bash

##########################################################################################################
# Multivalue Features - Direct MI computation without expansion
##########################################################################################################

# This example demonstrates using multivalue MI heuristics with OutRank.
# Multivalue features (e.g., "sports_music", "tech_art") are processed directly
# without expanding them into one-hot encoded binary features.

# IMPORTANT: Use '_' as delimiter in your CSV data for multivalue features!
# Example CSV format:
#   user_id,interests,skills,purchased,satisfaction
#   1,sports_music,python_sql,laptop_phone,high
#   2,music_tech,java_sql,phone_tablet,high

# Three multivalue MI algorithms are available:
# - MI-multivalue-set (recommended): Direct set-based mutual information
# - MI-multivalue-jaccard: Jaccard similarity-based approach
# - MI-multivalue-overlap: Overlap-based approach

# hint - if unsure what parameters do, you can always run "outrank --help"

outrank \
    --task all \
    --data_path examples/multivalue_data.csv \
    --data_source csv-raw \
    --heuristic MI-multivalue-set \
    --target_ranking_only True \
    --combination_number_upper_bound 2048 \
    --num_threads 8 \
    --output_folder ./ranking_outputs_multivalue \
    --subsampling 100

# Alternative: Use Jaccard-based multivalue MI
# outrank \
#     --task all \
#     --data_path examples/multivalue_data.csv \
#     --data_source csv-raw \
#     --heuristic MI-multivalue-jaccard \
#     --target_ranking_only True \
#     --combination_number_upper_bound 2048 \
#     --num_threads 8 \
#     --output_folder ./ranking_outputs_multivalue \
#     --subsampling 100

# Alternative: Use overlap-based multivalue MI
# outrank \
#     --task all \
#     --data_path examples/multivalue_data.csv \
#     --data_source csv-raw \
#     --heuristic MI-multivalue-overlap \
#     --target_ranking_only True \
#     --combination_number_upper_bound 2048 \
#     --num_threads 8 \
#     --output_folder ./ranking_outputs_multivalue \
#     --subsampling 100

