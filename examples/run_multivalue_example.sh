#!/bin/bash

##########################################################################################################
# Multivalue MI ranking
##########################################################################################################

# This run demonstrates multivalue MI computation with cardinality correction.
# Use '_' as delimiter in CSV for multivalue features (e.g., "sports_music").
# hint - if unsure what parameters do, you can always run "outrank --help"

outrank \
    --task all \
    --data_path examples/multivalue_data.csv \
    --data_source csv-raw \
    --heuristic MI-multivalue-set-randomized \
    --target_ranking_only True \
    --combination_number_upper_bound 2048 \
    --num_threads 8 \
    --output_folder ./ranking_outputs_multivalue \
    --subsampling 100
