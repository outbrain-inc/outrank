from __future__ import annotations

import glob
import json
import logging
import os
import signal
from typing import Any

import numpy as np
import pandas as pd
import gzip
import zstandard as zstd

from outrank.algorithms.importance_estimator import rank_features_3MR
from outrank.core_ranking import estimate_importances_minibatches
from outrank.core_ranking import estimate_importances_minibatches_hogwild
from outrank.core_utils import display_random_tip
from outrank.core_utils import display_tool_name
from outrank.core_utils import get_dataset_info
from outrank.core_utils import summarize_feature_bounds_for_transformers
from outrank.core_utils import summarize_rare_counts
from outrank.core_utils import write_json_dump_to_file

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
signal.signal(signal.SIGINT, signal.default_int_handler)

try:
    # pathos enables proper pickling during parallelization (multiprocessing does not)
    from pathos.multiprocessing import ProcessingPool as Pool

except Exception as es:
    logging.info(
        f'\U0001F631 Please install the "pathos" library (pip install pathos) for required multithreading capabilities. {es}',
    )


def outrank_task_conduct_ranking(args: Any) -> None:
    # Data source = folder structure + relevant file specifications
    if args.task in ['identify_rare_values', 'feature_summary_transformers']:
        args.heuristic = 'Constant'

    if args.disable_tqdm == 'False':
        display_tool_name()
        display_random_tip()

    dataset_info = get_dataset_info(args)

    for arg in vars(args):
        logging.info(f'{arg} set to: {getattr(args, arg)}')

    # Generate output folders (if not present)
    output_dir = os.path.dirname(
        os.path.join(
            args.output_folder, 'pairwise_ranks.tsv',
        ),
    )
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Initialize the global pool
    GLOBAL_CPU_POOL = Pool(args.num_threads)
    global_mutual_information_estimates = []
    global_bounds_storage = []
    global_memory_storage = []
    all_timings = []
    # Traverse the batches
    for raw_dump in glob.glob(dataset_info.data_path):

        if (
            args.data_source == 'ob-vw'
            or args.data_source == 'ob-csv'
            or args.data_source == 'csv-raw'
            or args.data_source == 'ob-raw-dump'
        ):
            all_subfiles = [raw_dump]

        for partial_data in all_subfiles:
            cmd_arguments = {
                'input_file': partial_data,
                'fw_col_mapping': dataset_info.fw_map,
                'column_descriptions': dataset_info.column_names,
                'numeric_column_types': dataset_info.column_types,
                'args': args,
                'data_encoding': dataset_info.encoding,
                'cpu_pool': GLOBAL_CPU_POOL,
                'delimiter': dataset_info.col_delimiter,
                'logger': logging,
            }

            if (
                args.data_source == 'ob-csv'
                or args.data_source == 'ob-vw'
                or args.data_source == 'csv-raw'
                or args.data_source == 'ob-raw-dump'
            ):
                # Choose between standard and Hogwild parallelism
                if args.enable_hogwild_parallelism == 'True':
                    logging.info('Using Hogwild-like parallelism for batch processing')
                    estimate_function = estimate_importances_minibatches_hogwild
                else:
                    estimate_function = estimate_importances_minibatches
                    
                (
                    checkpoint_timings,
                    mutual_information_estimates,
                    cardinality_object,
                    bounds_object_storage,
                    memory_object_storage,
                    coverage_object,
                    RARE_VALUE_STORAGE,
                    GLOBAL_PRIOR_COMB_COUNTS,
                    GLOBAL_ITEM_COUNTS,
                ) = estimate_function(**cmd_arguments)

            global_bounds_storage += bounds_object_storage
            global_memory_storage += memory_object_storage
            all_timings += checkpoint_timings

            if cardinality_object is None:
                continue

            if coverage_object is None:
                continue

            if mutual_information_estimates is not None:
                global_mutual_information_estimates.append(
                    mutual_information_estimates,
                )

    if args.task == 'identify_rare_values':
        logging.info('Summarizing rare values ..')
        summarize_rare_counts(
            RARE_VALUE_STORAGE, args, cardinality_object, dataset_info,
        )
        exit()

    if args.task == 'feature_summary_transformers':
        summarize_feature_bounds_for_transformers(
            bounds_object_storage,
            dataset_info.column_types,
            args.task,
            args.label_column,
        )
        exit()
    else:
        summary_of_numeric_features = summarize_feature_bounds_for_transformers(
            bounds_object_storage,
            dataset_info.column_types,
            args.task,
            args.label_column,
            output_summary_table_only=True,
        )
        if summary_of_numeric_features is not None:
            num_out = os.path.join(
                args.output_folder, 'numeric_feature_statistics.tsv',
            )
            summary_of_numeric_features.to_csv(num_out, sep='\t', index=False)
            logging.info(
                f'Stored statistics of numeric features to {num_out} ..',
            )

    # Just in case.
    GLOBAL_CPU_POOL.close()
    GLOBAL_CPU_POOL.join()

    if len(global_mutual_information_estimates) == 0:
        logging.info('No rankings were obtained, exiting ..')
        exit()

    # Compute median imps across batches
    triplets = pd.concat(global_mutual_information_estimates, axis=0)
    triplets.columns = ['FeatureA', 'FeatureB', 'Score']

    if '3mr' in args.heuristic:
        # relevance include MI-scores of features w.r.t. labels
        relevance_df = triplets[triplets.FeatureB == args.label_column].copy()
        relevance_df = relevance_df[
            relevance_df.FeatureA.map(lambda x: ' AND_REL ' not in x)
        ][['FeatureA', 'Score']]
        relevance_df = relevance_df[relevance_df.FeatureA != args.label_column]

        # relations include MI-scores of combinations w.r.t. label
        relations_df = triplets[triplets.FeatureB == args.label_column][
            ['FeatureA', 'Score']
        ].copy()
        relations_df = relations_df[
            relations_df.FeatureA.map(lambda x: ' AND_REL ' in x)
        ]
        relations_df['FeatureB'] = relations_df.FeatureA.map(
            lambda x: x.split(' AND_REL ')[1],
        )
        relations_df['FeatureA'] = relations_df.FeatureA.map(
            lambda x: x.split(' AND_REL ')[0],
        )

        # redundancies include MI-scores of features w.r.t. non-label features
        redundancies_df = triplets[(
            triplets.FeatureB != args.label_column
        )].copy()
        redundancies_df = redundancies_df[
            redundancies_df.FeatureA !=
            args.label_column
        ]
        redundancies_df = redundancies_df[
            redundancies_df.apply(
                lambda x: (' AND_REL ' not in x.FeatureA)
                and (' AND_REL ' not in x.FeatureB),
                axis=1,
            )
        ]

        # normalize
        relevance_df['score'] = (relevance_df.Score - relevance_df.Score.min()) / (
            relevance_df.Score.max() - relevance_df.Score.min()
        )
        relations_df['score'] = (relations_df.Score - relations_df.Score.min()) / (
            relations_df.Score.max() - relations_df.Score.min()
        )
        redundancies_df['score'] = (
            redundancies_df.Score - redundancies_df.Score.min()
        ) / (redundancies_df.Score.max() - redundancies_df.Score.min())

        # create dicts
        relevance_dict = {
            row.FeatureA: row.score for _,
            row in relevance_df.iterrows()
        }
        relations_dict = {
            (row.FeatureA, row.FeatureB): row.score
            for _, row in relations_df.iterrows()
        }
        relations_dict.update(
            {
                (row.FeatureB, row.FeatureA): row.score
                for _, row in relations_df.iterrows()
            },
        )
        redundancy_dict = {
            (row.FeatureA, row.FeatureB): row.score
            for _, row in redundancies_df.iterrows()
        }

        # compute 3mr ranks
        mrmrmr_ranking = rank_features_3MR(
            relevance_dict, redundancy_dict, relations_dict,
        )
        mrmrmr_ranking.to_csv(
            os.path.join(args.output_folder, '3mr_ranks.tsv'), sep='\t', index=False,
        )

    feature_first_modified = []
    feature_second_modified = []

    if args.include_cardinality_in_feature_names == 'True':
        for enx in range(triplets.shape[0]):
            feature_first = triplets.iloc[enx]['FeatureA']
            feature_second = triplets.iloc[enx]['FeatureB']
            card_first = str(len(cardinality_object[feature_first]))
            card_second = str(len(cardinality_object[feature_second]))
            cov_first = int(
                round((np.mean(np.array(coverage_object[feature_first]))), 1),
            )
            cov_second = int(
                round(np.mean(np.array(coverage_object[feature_second])), 1),
            )

            feature_first_modified.append(
                feature_first + f'-({card_first}; {cov_first})',
            )
            feature_second_modified.append(
                feature_second + f'-({card_second}; {cov_second})',
            )

        triplets['FeatureA'] = feature_first_modified
        triplets['FeatureB'] = feature_second_modified

    feature_memory_df = pd.DataFrame(global_memory_storage).mean()
    feature_memory_df.columns = ['NormalizedSize']
    feature_memory_df.to_csv(
        f'{args.output_folder}/memory.tsv', sep='\t', index=True,
    )

    triplets = triplets.sort_values(by=['Score'])

    triplets.to_csv(
        os.path.join(args.output_folder, 'pairwise_ranks.tsv'), sep='\t', index=False,
    )

    with open(f'{args.output_folder}/value_repetitions.json', 'w') as out_counts:
        out_dict = {}
        for k, v in GLOBAL_ITEM_COUNTS.items():
            actual_hist = np.array(list(v.default_counter.values()))
            more_than = lambda n, ary: len(np.where(ary > n)[0])
            out_dict[k] = {x: more_than(x, actual_hist)  for x in [0] + [1 * 10 ** x for x in range(6)]}
        out_counts.write(json.dumps(out_dict))

    with open(f'{args.output_folder}/combination_estimation_counts.json', 'w') as out_counts:
        out_dict = {str(k): v for k, v in GLOBAL_PRIOR_COMB_COUNTS.items()}
        out_counts.write(json.dumps(out_dict))

    # Write timings and config for replicability
    dfx = pd.DataFrame(all_timings)
    dfx.to_json(f'{args.output_folder}/timings.json')
    write_json_dump_to_file(args, f'{args.output_folder}/arguments.json')

    logging.info(
        f'Finished with ranking! Result stored as: {args.output_folder}/pairwise_ranks.tsv. Cleaning up tmp files ..',
    )

    os.remove('ranking_checkpoint_tmp.tsv')

def identify_data_file_type(data_path):
    all_files  = set(list(glob.glob(os.path.join(data_path, '*'))))
    gz_pname, zst_pname = 'data.vw.gz', 'data.vw.zst'
    if gz_pname in ''.join(all_files):
        return os.path.join(data_path, gz_pname)
    elif zst_pname in ''.join(all_files):
        return os.path.join(data_path, zst_pname)
    else:
        raise NotImplementedError('Please provide a valid data type .. (gz, zst)')
