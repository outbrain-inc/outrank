##########################################################################################################
# An optimization of generic OutRank invocation. It includes visualizations and other relevant statistics. #
##########################################################################################################

# This run compares features "one-at-a-time" and summarizes, visualizes the outputs.
# Heuristic is the same as MI-numba-randomized, but the code is optimized for faster executuion.
# hint - if unsure what parameters do, you can always run "outrank --help"

outrank \
    --task all \
    --data_path $PATH_TO_YOUR_DATA \
    --data_source csv-raw \
    --heuristic MI-numba-randomized-opt \
    --target_ranking_only True \
    --combination_number_upper_bound 2048 \
    --num_threads 12 \
    --interaction_order 1 \
    --output_folder ./some_output_folder \
    --subsampling 30
