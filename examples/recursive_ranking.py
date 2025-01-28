from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess

import pandas as pd

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger('syn-logger')

# Configuration constants
DATA_PATH = os.path.expanduser('~/datasets/blush')
MODEL_SPEC_DIR = 'model_spec_dir'
LABEL_COLUMN_NAME = 'label'
HEURISTIC = 'surrogate-SGD'
DATA_FORMAT = 'ob-vw'
NUM_THREADS = 6
INTERACTION_ORDER = 2
COMBINATION_NUMBER_BOUND = 1_000
MINIBATCH_SIZE = 10_000
SUBSAMPLING = 10

def run_outrank_task(reference_model_json: str, output_folder: str) -> None:
    """Run the outrank task with the specified parameters."""
    outrank_command = (
        f'outrank --task all --data_path {DATA_PATH} --data_source {DATA_FORMAT} '
        f'--target_ranking_only True --combination_number_upper_bound {COMBINATION_NUMBER_BOUND} '
        f'--num_threads {NUM_THREADS} --interaction_order {INTERACTION_ORDER} '
        f'--output_folder {output_folder} --reference_model_JSON {reference_model_json} '
        f'--heuristic {HEURISTIC} --label_column {LABEL_COLUMN_NAME} '
        f'--subsampling {SUBSAMPLING} --minibatch_size {MINIBATCH_SIZE} --disable_tqdm False;'
    )
    logger.info(f'Running outrank command: {outrank_command}')
    subprocess.run(outrank_command, shell=True, check=True)
    logger.info(f'Outrank task completed for {reference_model_json}')

def process_results(output_folder: str) -> str:
    """Read the results and extract the best feature."""
    results = pd.read_csv(os.path.join(output_folder, 'feature_singles.tsv'), delimiter='\t')
    best_feature = '-'.join(results.Feature.iloc[1].split('-')[:-1])
    best_feature = ','.join(best_feature.split(' AND '))
    logger.info(f'Best feature: {best_feature}')
    return best_feature

def update_model_spec(model_index: int, best_feature: str) -> None:
    """Update the model specification JSON with the new best feature."""
    current_model_path = os.path.join(MODEL_SPEC_DIR, f'model_{model_index}.json')
    next_model_path = os.path.join(MODEL_SPEC_DIR, f'model_{model_index + 1}.json')

    with open(current_model_path) as file:
        model_spec = json.load(file)

    current_features = model_spec['desc']['features']
    current_features.append(best_feature)
    logger.info(f'Updated features: {current_features}')

    with open(next_model_path, 'w') as file:
        new_model_spec = {'desc': {'features': current_features}}
        json.dump(new_model_spec, file)

def initialize_model_spec_dir() -> None:
    """Initialize the model specification directory with the initial JSON file."""
    command = (
        'mkdir -p model_spec_dir && '
        'rm -rv model_spec_dir/* && '
        'echo \'{"desc": {"features": []}}\' > ./model_spec_dir/model_0.json'
    )
    subprocess.run(command, shell=True, check=True)
    logger.info('Initialized model specification directory with model_0.json')

def run_evolution(iterations: int) -> None:
    """Main function to run the test for multiple iterations."""
    for i in range(iterations):
        reference_model_json = os.path.join(MODEL_SPEC_DIR, f'model_{i}.json')
        output_folder = f'output_dir_{i}'

        if os.path.isdir(output_folder):
            shutil.rmtree(output_folder)
        os.mkdir(output_folder)

        try:
            run_outrank_task(reference_model_json, output_folder)
            best_feature = process_results(output_folder)
            update_model_spec(i, best_feature)
        except Exception as e:
            logger.error(f'An error occurred during iteration {i}: {e}')
            continue

def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Run the outrank evolution process.')
    parser.add_argument(
        '--iterations',
        type=int,
        default=80,
        help='Number of iterations to run (default: 10)',
    )
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
    initialize_model_spec_dir()
    run_evolution(args.iterations)
