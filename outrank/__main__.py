from __future__ import annotations

import argparse
import logging

from outrank.cli_defaults import CLI_DEFAULTS
from outrank.task_generators import outrank_task_generate_data_set
from outrank.task_instance_ranking import outrank_task_rank_instances
from outrank.task_ranking import outrank_task_conduct_ranking
from outrank.task_selftest import conduct_self_test
from outrank.task_summary import outrank_task_result_summary
from outrank.task_visualization import outrank_task_visualize_results

logging.basicConfig(
    format='%(asctime)s - %(message)s',
    datefmt='%d-%b-%y %H:%M:%S',
)
logging.getLogger(__name__).setLevel(logging.INFO)

usage_examples = """
    Usage examples:

    # perform ranking, summary and visualize the results
    outrank --task all --data_path pathToSomeData --data_source ob-vw --heuristic MI-numba-randomized --include_cardinality_in_feature_names True --target_ranking_only True --combination_number_upper_bound 2048 --num_threads 8 --interaction_order 1 --transformers fw-transformers --output_folder ./ranking_outputs --subsampling 100

    # pairwise ranking only
    outrank --task ranking --data_path pathToSomeData --data_source ob-vw --heuristic MI-numba-randomized --target_ranking_only False --combination_number_upper_bound 10000 --num_threads 30 --output_folder ./ranking_outputs --subsampling 10

    # Higher order interactions
    outrank --task all --data_path pathToSomeData --data_source csv-raw --heuristic MI-numba-randomized --target_ranking_only True --combination_number_upper_bound 2048 --num_threads 8 --interaction_order 3 --output_folder ./ranking_outputs --subsampling 20

    # Using custom JSON transformers
    outrank --task ranking --data_path pathToSomeData --data_source csv-raw --heuristic MI-numba-randomized --transformers examples/custom_transformers.json --output_folder ./ranking_outputs

    # More docs and use cases at https://outbrain.github.io/outrank/outrank.html
"""

# Shorthand for default lookup — single source of truth in cli_defaults.py
D = CLI_DEFAULTS


def main():
    parser = argparse.ArgumentParser(
        description='Fast feature screening for sparse data sets.',
        epilog=usage_examples,
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        '--task',
        type=str,
        default=D['task'],
        help='Type of task to consider. Can be either "ranking", "ranking_summary", "feature_summary_transformers",  or "visualization"',
    )

    parser.add_argument(
        '--minibatch_size',
        type=int,
        default=D['minibatch_size'],
        help='Suitable for data, not pre-split to batches, this parameter determines batch size - note that too large batch size can slow down the multithreaded score computation due to many thread allocations etc. This works ok for <300 features and up to 48 threads.',
    )

    parser.add_argument(
        '--output_folder',
        type=str,
        default=D['output_folder'],
        help='Output folder containing ranking results.',
    )

    parser.add_argument(
        '--data_source',
        type=str,
        default=D['data_source'],
        help='Which database is used to obtain learning instances? this determines the inferred folder structure (csv-raw, ob-vw, ob-csv).',
    )

    parser.add_argument(
        '--data_path',
        type=str,
        default=D['data_path'],
        help='Path to the folder containing the main data used for subsequent learning.',
    )

    parser.add_argument(
        '--subsampling',
        type=int,
        default=D['subsampling'],
        help='Use every n-th row (e.g., 10 means ~10%% of data). Higher values = faster but less precise. Suggested: 10 to 100.',
    )

    parser.add_argument(
        '--combination_number_upper_bound',
        type=int,
        default=D['combination_number_upper_bound'],
        help='Cap the number of columns during feature ranking, per batch. This means that if you were to evaluate e.g., 100k combinations, this parameter results in behavior where only 2 ** 15 are taken into account (randomly) each bach, resulting in a monte-carlo like sampling scheme that yields estimates of the final ranks when all data is seen.',
    )

    parser.add_argument(
        '--missing_value_symbols',
        type=str,
        default=D['missing_value_symbols'],
        help='What symbols denote missing values? Comma-separate them - if comma is a missing symbol itself please open an issue.',
    )

    parser.add_argument(
        '--heuristic',
        type=str,
        default=D['heuristic'],
        help='Selected heuristic (that performs feature scoring). For full list please see the docs: https://outbrain.github.io/outrank/outrank/algorithms/importance_estimator.html',
    )

    parser.add_argument(
        '--include_noise_baseline_features',
        type=str,
        default=D['include_noise_baseline_features'],
        help='If enabled, it computes five control variables (random noises)',
    )

    parser.add_argument(
        '--include_cardinality_in_feature_names',
        type=str,
        default=D['include_cardinality_in_feature_names'],
        help='If enabled, feature names appear as feature-(cardinality) for easier inspection/debugging.',
    )

    parser.add_argument(
        '--image_format',
        type=str,
        default=D['image_format'],
        help='The format of the output images (task: visualization)',
    )

    parser.add_argument(
        '--num_threads', type=int, default=D['num_threads'], help='Number of threads to consider. More threads implies faster ranking, however, there will be some memory overhead. Should be as large as the machine can handle memory-wise.',
    )

    parser.add_argument(
        '--label_column',
        type=str,
        default=D['label_column'],
        help='Name of the target attribute for ranking. Note that this can be any other feature for most implemented heuristics.',
    )

    parser.add_argument(
        '--max_unique_hist_constraint',
        type=int,
        default=D['max_unique_hist_constraint'],
        help='Max number of unique values for which counts are recalled.',
    )

    parser.add_argument(
        '--transformers',
        type=str,
        default=D['transformers'],
        help='Collection of which feature transformations to consider. Examples are: fw-transformers, default, minimal. Also supports JSON file paths (e.g., custom_transformers.json) and combinations (e.g., default,custom.json)',
    )

    parser.add_argument(
        '--rare_value_count_upper_bound',
        type=int,
        default=D['rare_value_count_upper_bound'],
        help="When identifying rare attr-val pairs, what's the upper frequency bound?",
    )

    parser.add_argument(
        '--feature_set_focus',
        type=str,
        default=D['feature_set_focus'],
        help='Collection of which feature transformations to consider',
    )

    parser.add_argument(
        '--interaction_order',
        type=int,
        default=D['interaction_order'],
        help='The order of feature interactions to consider during ranking (complex features comprised of n elementary ones)',
    )

    parser.add_argument(
        '--reference_model_JSON',
        type=str,
        default=D['reference_model_JSON'],
        help='Reference model JSON',
    )

    parser.add_argument(
        '--target_ranking_only',
        type=str,
        default=D['target_ranking_only'],
        help='Compute only the feature-label scores? This is substantially faster (O(n)).',
    )

    parser.add_argument(
        '--explode_multivalue_features',
        type=str,
        default=D['explode_multivalue_features'],
        help="Which ';'-separated features should be one-hot encoded into n new features (coverage analysis)",
    )

    parser.add_argument(
        '--subfeature_mapping',
        type=str,
        default=D['subfeature_mapping'],
        help='Compute sub-features on-the fly. Example: featureA->featureB implies features based on each value of featureA will be considered. So, feature names will correspond to values of the first feature, with actual values being constructed based on the second feature (two or more possible values).',
    )

    parser.add_argument(
        '--num_synthetic_features',
        type=int,
        default=D['num_synthetic_features'],
        help='Relevant for task data_generator -- how many features.',
    )

    parser.add_argument(
        '--tldr',
        type=str,
        default=D['tldr'],
        help='If enabled, it will output some of the main results on the screen after finishing.',
    )

    parser.add_argument(
        '--num_synthetic_rows',
        type=int,
        default=D['num_synthetic_rows'],
        help='Relevant for task data_generator -- how many rows.',
    )

    parser.add_argument(
        '--generator_type',
        type=str,
        default=D['generator_type'],
        help='Relevant for task data_generator -- which generator to consider',
    )

    parser.add_argument(
        '--output_synthetic_df_name',
        type=str,
        default=D['output_synthetic_df_name'],
        help='Relevant for task data_generator -- name of the folder that contains generated data.',
    )

    parser.add_argument(
        '--disable_tqdm',
        default=D['disable_tqdm'],
        choices=['False', 'True'],
        help='Either True or False.',
    )

    parser.add_argument(
        '--mi_stratified_sampling_ratio',
        type=float,
        default=D['mi_stratified_sampling_ratio'],
        help='If < 1.0, MI algorithm will further subsample data in stratified manner (equal distributions per value if possible).',
    )

    parser.add_argument(
        '--compute_jmi',
        choices=['False', 'True'],
        default=D['compute_jmi'],
        help='If True, compute JMI (Joint Mutual Information) greedy feature selection after pairwise ranking. Detects XOR-like synergies missed by pairwise MI.',
    )

    parser.add_argument(
        '--jmi_top_k',
        type=int,
        default=D['jmi_top_k'],
        help='Number of top pairwise-MI features to consider for JMI ranking (The Screening Rule). Higher = more thorough but slower (O(k^2) CMI calls).',
    )

    parser.add_argument(
        '--compute_interaction_info',
        choices=['False', 'True'],
        default=D['compute_interaction_info'],
        help='If True, compute interaction information II(X_i,X_j;Y) for top feature pairs. Negative II = synergy, positive = redundancy.',
    )

    parser.add_argument(
        '--interaction_info_top_k',
        type=int,
        default=D['interaction_info_top_k'],
        help='Number of top features to compute pairwise interaction information for.',
    )

    args = parser.parse_args()

    if args.task == 'selftest':
        conduct_self_test('MI-numba-randomized')
        exit()

    if args.data_path is None and args.task != 'data_generator':
        logging.error('Please specify data set name (--data_path).')
        exit()

    all_tasks_to_consider = []
    if args.task != 'all':
        all_tasks_to_consider = [args.task]

    else:
        all_tasks_to_consider = ['ranking', 'ranking_summary', 'visualization']

    for task in all_tasks_to_consider:
        logging.info(f'Proceeding with task: {task} ..')

        if (
            task == 'ranking'
            or task == 'feature_summary_transformers'
            or task == 'identify_rare_values'
        ):
            outrank_task_conduct_ranking(args)

        elif task == 'visualization':
            outrank_task_visualize_results(args)

        elif task == 'ranking_summary':
            outrank_task_result_summary(args)

        elif task == 'data_generator':
            outrank_task_generate_data_set(args)

        elif task == 'instance_ranking':
            outrank_task_rank_instances(args)

        else:
            logging.info(f'Warning, the selected task: {task} does not exist.')


if __name__ == '__main__':
    main()
