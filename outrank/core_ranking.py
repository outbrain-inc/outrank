from __future__ import annotations

import gzip
import itertools
import logging
import os
import random
import time
from collections import Counter
from collections import defaultdict
from collections import deque
from timeit import default_timer as timer
from typing import Any
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import queue

import numpy as np
import pandas as pd
import tqdm
import xxhash
import zstandard as zstd

from outrank.algorithms.importance_estimator import \
    get_importances_estimate_pairwise
from outrank.algorithms.sketches.counting_counters_ordinary import \
    PrimitiveConstrainedCounter
from outrank.algorithms.sketches.counting_ultiloglog import \
    HyperLogLogWCache as HyperLogLog
from outrank.core_utils import BatchRankingSummary
from outrank.core_utils import extract_features_from_reference_JSON
from outrank.core_utils import generic_line_parser
from outrank.core_utils import get_num_of_instances
from outrank.core_utils import internal_hash
from outrank.core_utils import is_prior_heuristic
from outrank.core_utils import NominalFeatureSummary
from outrank.core_utils import NumericFeatureSummary
from outrank.feature_transformations.ranking_transformers import FeatureTransformerGeneric
from outrank.feature_transformations.ranking_transformers import FeatureTransformerNoise

logger = logging.getLogger('syn-logger')
logger.setLevel(logging.DEBUG)
random.seed(a=123, version=2)
GLOBAL_CARDINALITY_STORAGE: dict[Any, Any] = dict()
GLOBAL_COUNTS_STORAGE: dict[Any, Any] = dict()
GLOBAL_RARE_VALUE_STORAGE: dict[str, Any] = Counter()
GLOBAL_PRIOR_COMB_COUNTS: dict[Any, int] = Counter()
IGNORED_VALUES = set()
HYPERLL_ERROR_BOUND = 0.02
MAX_FEATURES_3MR = 10 ** 4


def prior_combinations_sample(combinations: list[tuple[Any, ...]], args: Any) -> list[tuple[Any, ...]]:
    """Make sure only relevant subspace of combinations is selected based on prior counts"""

    if len(combinations) == 0:
        return []

    missing_combinations = set(set(combinations)).difference(GLOBAL_PRIOR_COMB_COUNTS.keys())
    if len(missing_combinations) > 0:
        for combination in missing_combinations:
            GLOBAL_PRIOR_COMB_COUNTS[combination] = 0

    tmp = sorted(combinations, key=GLOBAL_PRIOR_COMB_COUNTS.get, reverse=False)[:args.combination_number_upper_bound]

    for combination in tmp:
        GLOBAL_PRIOR_COMB_COUNTS[combination] += 1

    return tmp


def get_combinations_from_columns(all_columns: pd.Index, args: Any) -> list[tuple[Any, ...]]:
    """Return feature-feature & feature-label combinations, depending on the heuristic and ranking scope"""

    if '3mr' in args.heuristic:
        if args.combination_number_upper_bound > MAX_FEATURES_3MR:
            args.combination_number_upper_bound = MAX_FEATURES_3MR
        rel_columns = [column for column in all_columns if ' AND_REL ' in column]
        non_rel_columns = sorted(set(all_columns) - set(rel_columns))

        combinations = list(
            itertools.combinations_with_replacement(non_rel_columns, 2),
        )
        combinations += [(column, args.label_column) for column in rel_columns]
    else:
        _combinations = itertools.combinations_with_replacement(all_columns, 2)

        # Some applications do not require the full feature-feature triangular matrix
        if args.target_ranking_only == 'True':
            combinations = [x for x in _combinations if args.label_column in x]
        else:
            combinations = list(_combinations)

    if args.target_ranking_only != 'True':
        # Diagonal elements (non-label)
        combinations += [
            (individual_column, individual_column)
            for individual_column in all_columns
            if individual_column != args.label_column
        ]
    return combinations


def mixed_rank_graph(
    input_dataframe: pd.DataFrame, args: Any, cpu_pool: Any, pbar: Any,
) -> BatchRankingSummary:
    """Compute the full mixed rank graph corresponding to all pairwise feature interactions based on the selected heuristic"""

    all_columns = input_dataframe.columns

    triplets = []
    tmp_df = input_dataframe.copy().astype('category')
    out_time_struct = {}

    # Handle cont. types prior to interaction evaluation
    pbar.set_description('Encoding columns')
    start_enc_timer = timer()
    tmp_df = pd.DataFrame({k : tmp_df[k].cat.codes for k in all_columns})

    end_enc_timer = timer()
    out_time_struct['encoding_columns'] = end_enc_timer - start_enc_timer

    combinations = get_combinations_from_columns(all_columns, args)

    reference_model_features = {}
    if is_prior_heuristic(args):
        reference_model_features = [(' AND ').join(tuple(sorted(item.split(',')))) for item in extract_features_from_reference_JSON(args.reference_model_JSON, all_features=True)]
        combinations = [comb for comb in combinations if comb[0] not in reference_model_features and comb[1] not in reference_model_features]

    combinations = prior_combinations_sample(combinations, args)
    random.shuffle(combinations)

    if args.heuristic == 'Constant':
        final_constant_imp = []
        for c1, c2 in combinations:
            final_constant_imp.append((c1, c2, 0.0))

        out_time_struct['feature_score_computation'] = end_enc_timer - \
            start_enc_timer
        return BatchRankingSummary(final_constant_imp, out_time_struct)

    # Map the scoring calls to the worker pool
    pbar.set_description('Allocating thread pool')

    # starmap is an alternative that is slower unfortunately (but nicer)
    def get_grounded_importances_estimate(combination: tuple[str]) -> Any:
        return get_importances_estimate_pairwise(combination, reference_model_features, args, tmp_df=tmp_df)

    start_enc_timer = timer()
    with cpu_pool as p:
        pbar.set_description(f'Computing (#ftr={len(combinations)})')
        results = p.amap(get_grounded_importances_estimate, combinations)
        while not results.ready():
            time.sleep(4)
        triplets = results.get()
    end_enc_timer = timer()
    out_time_struct['feature_score_computation'] = end_enc_timer - \
        start_enc_timer

    # Gather the final triplets
    pbar.set_description('Aggregation of ranking results')
    final_triplets = []
    for triplet in triplets:
        inv = (triplet[1], triplet[0], triplet[2])
        final_triplets.append(inv)
        final_triplets.append(triplet)
        triplets = final_triplets

    pbar.set_description('Proceeding to the next batch of data')
    return BatchRankingSummary(triplets, out_time_struct)


def enrich_with_transformations(
    input_dataframe: pd.DataFrame, num_col_types: set[str], logger: Any, args: Any,
) -> pd.DataFrame:
    """Construct a collection of new features based on pre-defined transformations/rules"""

    transformer = FeatureTransformerGeneric(
        num_col_types, preset=args.transformers,
    )
    transformed_df = transformer.construct_new_features(input_dataframe)
    logger.info(
        f'Constructed {len(transformer.constructed_feature_names)} new features ..',
    )

    return transformed_df


def compute_combined_features(
    input_dataframe: pd.DataFrame,
    args: Any,
    pbar: Any,
    is_3mr: bool = False,
) -> pd.DataFrame:
    """Compute higher order features via xxhash-based trick."""

    all_columns = [
        x for x in input_dataframe.columns if x != args.label_column
    ]
    join_string = ' AND_REL ' if is_3mr else ' AND '
    interaction_order = 2 if is_3mr else args.interaction_order

    full_combination_space = []

    if args.interaction_order > 1:
        full_combination_space = list(
            itertools.combinations(all_columns, interaction_order),
        )
    full_combination_space = prior_combinations_sample(full_combination_space, args)

    if args.reference_model_JSON != '':
        model_combinations = extract_features_from_reference_JSON(args.reference_model_JSON, combined_features_only=True)
        model_combinations = [tuple(sorted(combination.split(','))) for combination in model_combinations]
        if not is_prior_heuristic(args):
            full_combination_space = model_combinations

    if is_prior_heuristic(args):
        full_combination_space = full_combination_space + [tuple for tuple in model_combinations if tuple not in full_combination_space]

    def combine_features(new_combination):
        combined_feature = input_dataframe[new_combination[0]].astype(str)
        for feature in new_combination[1:]:
            combined_feature += input_dataframe[feature].astype(str)
        combined_feature = combined_feature.apply(lambda x: xxhash.xxh64(x).hexdigest())
        ftr_name = join_string.join(new_combination)
        return ftr_name, combined_feature

    new_feature_hash = {}
    for idx, new_combination in enumerate(full_combination_space):
        pbar.set_description(f'Created {idx + 1}/{len(full_combination_space)}')
        ftr_name, combined_feature = combine_features(new_combination)
        new_feature_hash[ftr_name] = combined_feature

    tmp_df = pd.DataFrame(new_feature_hash)
    pbar.set_description('Concatenating into final frame ..')
    input_dataframe = pd.concat([input_dataframe, tmp_df], axis=1)
    del tmp_df

    return input_dataframe


def compute_expanded_multivalue_features(
    input_dataframe: pd.DataFrame, logger: Any, args: Any, pbar: Any,
) -> pd.DataFrame:
    """Compute one-hot encoded feature space based on each designated multivalue feature. E.g., feature with value "a,b,c" becomes three features, values of which are presence of a given value in a mutlivalue feature of choice."""

    considered_multivalue_features = args.explode_multivalue_features.split(
        ';',
    )
    new_feature_hash = {}
    missing_symbols = set(args.missing_value_symbols.split(','))

    for multivalue_feature in considered_multivalue_features:
        multivalue_feature_vector = input_dataframe[multivalue_feature].values.tolist(
        )
        multivalue_feature_vector = [
            x.replace(',', '-') for x in multivalue_feature_vector
        ]
        multivalue_sets = [
            set(x.split('-'))
            for x in multivalue_feature_vector
        ]
        unique_values = set.union(*multivalue_sets)

        for missing_symbol in missing_symbols:
            if missing_symbol in unique_values:
                unique_values.remove(missing_symbol)

        for unique_value in unique_values:
            tmp_vec = []
            for enx, multivalue in enumerate(multivalue_sets):
                if unique_value in multivalue:
                    tmp_vec.append('1')
                else:
                    tmp_vec.append('')

            new_feature_hash[f'MULTIEX-{multivalue_feature}-{unique_value}'] = tmp_vec

    tmp_df = pd.DataFrame(new_feature_hash)
    input_dataframe = pd.concat([input_dataframe, tmp_df], axis=1)
    del tmp_df

    return input_dataframe


def compute_subfeatures(
    input_dataframe: pd.DataFrame, logger: Any, args: Any, pbar: Any,
) -> pd.DataFrame:
    """Compute derived features that are more fine-grained. Implements logic around two operators that govern feature construction.
    ->: One sided construction - every value from left side is fine, separate ones from the right side feature will be considered.
    <->: Two sided construction - two-sided values present. This means that each value from a is combined with each from b, forming |A|*|B| new features (one-hot encoded)
    """

    all_subfeature_pair_seeds = args.subfeature_mapping.split(';')
    new_feature_hash = dict()

    for seed_pair in all_subfeature_pair_seeds:
        if '<->' in seed_pair:
            feature_first, feature_second = seed_pair.split('<->')

        elif '->' in seed_pair:
            feature_first, feature_second = seed_pair.split('->')

        else:
            raise NotImplementedError(
                'Please specify valid subfeature operator (<-> or ->)',
            )

        subframe = input_dataframe[[feature_first, feature_second]]
        unique_feature_second = subframe[feature_second].unique()
        feature_first_vec = subframe[feature_first].tolist()
        feature_second_vec = subframe[feature_second].tolist()
        out_template_feature = [
            (a, b) for a, b in zip(feature_first_vec, feature_second_vec)
        ]

        if '<->' in seed_pair:
            unique_feature_first = subframe[feature_first].unique()

            mask_types = []
            for unique_target_feature_value in unique_feature_second:
                for unique_seed_feature_value in unique_feature_first:
                    mask_types.append(
                        (unique_seed_feature_value, unique_target_feature_value),
                    )

            for mask_type in mask_types:
                new_feature = []
                for value_tuple in out_template_feature:
                    if (
                        value_tuple[0] == mask_type[0]
                        and value_tuple[1] == mask_type[1]
                    ):
                        new_feature.append(str(1))
                    else:
                        new_feature.append(str(0))
                feature_name = (
                    f'SUBFEATURE|{feature_first}|{feature_second}-'
                    + mask_type[0]
                    + '&'
                    + mask_type[1]
                )
                new_feature_hash[feature_name] = new_feature

            del new_feature

        elif '->' in seed_pair:
            for unique_target_feature_value in unique_feature_second:
                tmp_new_feature = [
                    'AND'.join(
                        x,
                    ) if x[1] == unique_target_feature_value else ''
                    for x in out_template_feature
                ]
                feature_name_final = (
                    'SUBFEATURE-' + feature_first + '&' + unique_target_feature_value
                )
                new_feature_hash[feature_name_final] = tmp_new_feature

    tmp_df = pd.DataFrame(new_feature_hash)
    input_dataframe = pd.concat([input_dataframe, tmp_df], axis=1)

    del tmp_df
    return input_dataframe


def include_noisy_features(
    input_dataframe: pd.DataFrame, logger: Any, args: Any,
) -> pd.DataFrame:
    """Add randomized features that serve as a sanity check"""

    transformer = FeatureTransformerNoise()
    transformed_df = transformer.construct_new_features(
        input_dataframe, args.label_column,
    )

    return transformed_df


def compute_coverage(input_dataframe: pd.DataFrame, args: Any) -> dict[str, set[str]]:
    """Compute coverage of features, incrementally"""
    output_storage_cov = defaultdict(set)
    all_missing_symbols = set(args.missing_value_symbols.split(','))
    for column in input_dataframe:
        all_missing = sum(
            [
                input_dataframe[column].values.tolist().count(x)
                for x in all_missing_symbols
            ],
        )

        output_storage_cov[column] = (
            1 - (all_missing / input_dataframe.shape[0])
        ) * 100

    return output_storage_cov


def compute_feature_memory_consumption(input_dataframe: pd.DataFrame, args: Any) -> dict[str, set[str]]:
    """An approximation of how much feature take up"""
    output_storage_features = defaultdict(set)
    for col in input_dataframe.columns:
        specific_column = [
            str(x).strip() for x in input_dataframe[col].astype(str).values.tolist()
        ]
        col_size = sum(
            len(x.encode())
            for x in specific_column
        ) / input_dataframe.shape[0]
        output_storage_features[col] = col_size
    return output_storage_features


def compute_value_counts(input_dataframe: pd.DataFrame, args: Any):
    """Update the count structure"""

    global GLOBAL_RARE_VALUE_STORAGE
    global IGNORED_VALUES

    ignored_values = IGNORED_VALUES
    global_storage = GLOBAL_RARE_VALUE_STORAGE
    rare_value_count_upper_bound = args.rare_value_count_upper_bound

    for column in input_dataframe.columns:
        main_values = input_dataframe[column].values
        for value in main_values:
            if value not in ignored_values:
                global_storage[(column, value)] += 1

    keys_to_remove = []
    for key, val in global_storage.items():
        if val > rare_value_count_upper_bound:
            ignored_values.add(key)
            keys_to_remove.append(key)

    for key in keys_to_remove:
        del global_storage[key]

    # Update global variables
    GLOBAL_RARE_VALUE_STORAGE = global_storage
    IGNORED_VALUES = ignored_values


def compute_cardinalities(input_dataframe: pd.DataFrame, pbar: Any, max_unique_hist_constraint: int) -> None:
    global GLOBAL_CARDINALITY_STORAGE
    global GLOBAL_COUNTS_STORAGE

    output_storage_card = defaultdict(set)
    for enx, column in enumerate(input_dataframe.columns):
        column_data = input_dataframe[column]
        unique_values = set(column_data)
        output_storage_card[column] = unique_values

        if column not in GLOBAL_CARDINALITY_STORAGE:
            GLOBAL_CARDINALITY_STORAGE[column] = HyperLogLog(HYPERLL_ERROR_BOUND)

        if column not in GLOBAL_COUNTS_STORAGE:
            GLOBAL_COUNTS_STORAGE[column] = PrimitiveConstrainedCounter(max_unique_hist_constraint)

        for value in column_data.values:
            GLOBAL_COUNTS_STORAGE[column].add(value)

        for unique_value in unique_values:
            if unique_value:
                GLOBAL_CARDINALITY_STORAGE[column].add(internal_hash(unique_value))

        pbar.set_description(f'Computing cardinality (Hyperloglog update) {enx+1}/{input_dataframe.shape[1]}')


def compute_bounds_increment(
    input_dataframe: pd.DataFrame, numeric_column_types: set[str],
) -> dict[str, Any]:
    numeric_column_types = set(numeric_column_types)
    summary_object = {}

    for feature in input_dataframe.columns:
        feature_vector = input_dataframe[feature]
        if feature in numeric_column_types:
            feature_vector = pd.to_numeric(feature_vector, errors='coerce')
            summary_object[feature] = NumericFeatureSummary(
                feature,
                np.min(feature_vector),
                np.max(feature_vector),
                np.mean(feature_vector),
                len(np.unique(feature_vector)),
            )
        else:
            summary_object[feature] = NominalFeatureSummary(
                feature,
                len(np.unique(feature_vector)),
            )

    return summary_object


def compute_batch_ranking(
    line_tmp_storage: list[list[Any]],
    numeric_column_types: set[str],
    args: Any,
    cpu_pool: Any,
    column_descriptions: list[str],
    logger: Any,
    pbar: Any,
) -> tuple[
    BatchRankingSummary, dict[str, Any], dict[str, set[str]], dict[str, set[str]],
]:
    """Enrich the feature space and compute the batch importances"""

    input_dataframe = pd.DataFrame(line_tmp_storage, columns=column_descriptions)
    pbar.set_description('Control features')

    if args.feature_set_focus:
        focus_set = set()
        if args.feature_set_focus == '_all_from_reference_JSON':
            focus_set = extract_features_from_reference_JSON(args.reference_model_JSON)
        else:
            focus_set = set(args.feature_set_focus.split(','))

        focus_set.add(args.label_column)
        focus_set = {x for x in focus_set if x in input_dataframe.columns}
        input_dataframe = input_dataframe[list(focus_set)]

    if args.transformers != 'none':
        pbar.set_description('Adding transformations')
        input_dataframe = enrich_with_transformations(
            input_dataframe, numeric_column_types, logger, args,
        )

    if args.explode_multivalue_features != 'False':
        pbar.set_description('Constructing new features from multivalue ones')
        input_dataframe = compute_expanded_multivalue_features(
            input_dataframe, logger, args, pbar,
        )

    if args.subfeature_mapping != 'False':
        pbar.set_description('Constructing new (sub)features')
        input_dataframe = compute_subfeatures(input_dataframe, logger, args, pbar)

    if args.interaction_order > 1 or args.reference_model_JSON:
        pbar.set_description('Constructing new features')
        input_dataframe = compute_combined_features(input_dataframe, args, pbar)

    if '3mr' in args.heuristic:
        pbar.set_description('Constructing features for computing relations in 3mr')
        input_dataframe = compute_combined_features(
            input_dataframe, args, pbar, True,
        )

    if args.include_noise_baseline_features == 'True' and args.heuristic != 'Constant':
        pbar.set_description('Computing baseline features')
        input_dataframe = include_noisy_features(input_dataframe, logger, args)

    pbar.set_description('Computing coverage')
    coverage_storage = compute_coverage(input_dataframe, args)
    feature_memory_consumption = compute_feature_memory_consumption(
        input_dataframe, args,
    )
    compute_cardinalities(input_dataframe, pbar, args.max_unique_hist_constraint)

    if args.task == 'identify_rare_values':
        compute_value_counts(input_dataframe, args)

    bounds_storage = compute_bounds_increment(input_dataframe, numeric_column_types)

    pbar.set_description(
        f'Computing ranks for {input_dataframe.shape[1]} features',
    )

    return (
        mixed_rank_graph(input_dataframe, args, cpu_pool, pbar),
        bounds_storage,
        coverage_storage,
        feature_memory_consumption,
    )


def get_grouped_df(importances_df_list: list[tuple[str, str, float]]) -> pd.DataFrame:
    """A helper method that enables median-based aggregation after processing"""

    importances_df = pd.DataFrame(importances_df_list, columns=['FeatureA', 'FeatureB', 'Score'])
    if importances_df.empty:
        return None
    grouped = importances_df.groupby(['FeatureA', 'FeatureB'], as_index=False).median()

    return grouped


def checkpoint_importances_df(importances_batch: list[tuple[str, str, float]]) -> None:
    """A helper which stores intermediary state - useful for longer runs"""

    gdf = get_grouped_df(importances_batch)
    if gdf is not None:
        gdf.to_csv('ranking_checkpoint_tmp.tsv', sep='\t')


def estimate_importances_minibatches_hogwild(
    input_file: str,
    column_descriptions: list,
    fw_col_mapping: dict[str, str],
    numeric_column_types: set,
    batch_size: int = 100000,
    args: Any = None,
    data_encoding: str = 'utf-8',
    cpu_pool: Any = None,
    delimiter: str = '\t',
    feature_construction_mode: bool = False,
    logger: Any = None,
) -> tuple[list[dict[str, Any]], Any, dict[Any, Any], list[dict[str, Any]], list[dict[str, set[str]]], defaultdict[str, list[set[str]]], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Hogwild-like interaction score estimator where idle threads can start processing subsequent batches.
    
    This implementation allows concurrent batch processing with overwriting capabilities for improved 
    resource utilization, especially beneficial for higher-order interactions.
    """
    
    invalid_line_queue: Any = deque([], maxlen=2**5)
    invalid_lines = 0
    line_counter = 0
    
    # Thread-safe storage for results
    importances_df: list[Any] = []
    bounds_storage_batch = []
    memory_storage_batch = []
    step_timing_checkpoints = []
    
    local_coverage_object = defaultdict(list)
    
    # Thread synchronization objects
    results_lock = threading.Lock()
    batch_queue = queue.Queue(maxsize=args.num_threads * 2)  # Buffer for batches
    
    local_pbar = tqdm.tqdm(
        total=get_num_of_instances(input_file) - 1, position=0, disable=args.disable_tqdm == 'True',
    )
    
    file_name, file_extension = os.path.splitext(input_file)
    
    if file_extension == '.gz':
        file_stream = gzip.open(input_file, 'rt', encoding=data_encoding)
    elif file_extension == '.zst':
        file_stream = zstd.open(input_file, 'rt', encoding=data_encoding)
    else:
        file_stream = open(input_file, encoding=data_encoding)
    
    file_stream.readline()  # Skip header
    
    def process_batch_worker(batch_data, batch_id):
        """Worker function to process a single batch"""
        try:
            importances_batch, bounds_storage, coverage_storage, memory_storage = compute_batch_ranking(
                batch_data,
                numeric_column_types,
                args,
                cpu_pool,
                column_descriptions,
                logger,
                local_pbar,
            )
            
            # Thread-safe result aggregation with potential overwriting
            with results_lock:
                # In Hogwild fashion, later batches can overwrite earlier ones
                # We'll merge results, with later batch_id taking precedence
                
                # Store results with batch identifier for potential overwriting
                result_entry = {
                    'batch_id': batch_id,
                    'importances': importances_batch,
                    'bounds': bounds_storage,
                    'coverage': coverage_storage,
                    'memory': memory_storage
                }
                
                return result_entry
                
        except Exception as e:
            logger.error(f"Error processing batch {batch_id}: {e}")
            return None
    
    # Read data and create batches
    line_tmp_storage = []
    batch_id = 0
    batch_futures = []
    
    local_pbar.set_description('Starting Hogwild ranking computation')
    
    # Use ThreadPoolExecutor for Hogwild-like concurrent processing
    with ThreadPoolExecutor(max_workers=args.num_threads) as executor:
        
        for line in file_stream:
            line_counter += 1
            local_pbar.update(1)
            
            if line_counter % args.subsampling != 0:
                continue
            
            parsed_line = generic_line_parser(
                line, delimiter, args, fw_col_mapping, column_descriptions,
            )
            
            if len(parsed_line) == len(column_descriptions):
                line_tmp_storage.append(parsed_line)
            else:
                invalid_line_queue.appendleft(str(parsed_line))
                invalid_lines += 1
            
            # Submit batch for processing when ready
            if len(line_tmp_storage) >= args.minibatch_size:
                # Submit the batch to be processed concurrently
                future = executor.submit(process_batch_worker, line_tmp_storage.copy(), batch_id)
                batch_futures.append(future)
                
                line_tmp_storage = []
                batch_id += 1
                
                local_pbar.set_description(f'Submitted batch {batch_id} for Hogwild processing')
        
        # Process remaining data if any
        if line_tmp_storage and len(line_tmp_storage) > 2**10:
            line_tmp_storage = line_tmp_storage[:args.minibatch_size]
            future = executor.submit(process_batch_worker, line_tmp_storage, batch_id)
            batch_futures.append(future)
        
        file_stream.close()
        
        # Collect results in Hogwild fashion - as they complete
        local_pbar.set_description('Collecting Hogwild results')
        
        completed_batches = {}
        for future in as_completed(batch_futures):
            result = future.result()
            if result is not None:
                batch_id = result['batch_id']
                completed_batches[batch_id] = result
                
                # Hogwild behavior: immediately integrate results, allowing overwriting
                importances_df += result['importances'].triplet_scores
                bounds_storage_batch.append(result['bounds'])
                memory_storage_batch.append(result['memory'])
                step_timing_checkpoints.append(result['importances'].step_times)
                
                for k, v in result['coverage'].items():
                    local_coverage_object[k].append(v)
                
                if args.heuristic != 'Constant':
                    checkpoint_importances_df(importances_df)
                
                local_pbar.set_description(f'Processed batch {batch_id} in Hogwild mode')
    
    local_pbar.set_description('Wrapping up Hogwild processing')
    local_pbar.close()
    
    if invalid_lines > 0:
        logger.info(
            f"Detected {invalid_lines} invalid lines. If this number is very high, it's possible your header is off - re-check your data/attribute-feature mappings please!",
        )
        
        invalid_lines_log = '\n INVALID_LINE ====> '.join(
            list(invalid_line_queue)[0:5],
        )
        logger.info(
            f'5 samples of invalid lines are printed below\n {invalid_lines_log}',
        )
    
    return (
        step_timing_checkpoints,
        get_grouped_df(importances_df),
        GLOBAL_CARDINALITY_STORAGE.copy(),
        bounds_storage_batch,
        memory_storage_batch,
        local_coverage_object,
        GLOBAL_RARE_VALUE_STORAGE.copy(),
        GLOBAL_PRIOR_COMB_COUNTS.copy(),
        GLOBAL_COUNTS_STORAGE.copy(),
    )


def estimate_importances_minibatches(
    input_file: str,
    column_descriptions: list,
    fw_col_mapping: dict[str, str],
    numeric_column_types: set,
    batch_size: int = 100000,
    args: Any = None,
    data_encoding: str = 'utf-8',
    cpu_pool: Any = None,
    delimiter: str = '\t',
    feature_construction_mode: bool = False,
    logger: Any = None,
) -> tuple[list[dict[str, Any]], Any, dict[Any, Any], list[dict[str, Any]], list[dict[str, set[str]]], defaultdict[str, list[set[str]]], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Interaction score estimator - suitable for example for csv-like input data types.
    This type of data is normally a single large csv, meaning that minibatch processing needs to
    happen during incremental handling of the file (that"s not the case for pre-separated ob data)
    """

    invalid_line_queue: Any = deque([], maxlen=2**5)

    invalid_lines = 0
    line_counter = 0

    importances_df: list[Any] = []
    line_tmp_storage = []
    bounds_storage_batch = []
    memory_storage_batch = []
    step_timing_checkpoints = []

    local_coverage_object = defaultdict(list)
    local_pbar = tqdm.tqdm(
        total=get_num_of_instances(input_file) - 1, position=0, disable=args.disable_tqdm == 'True',
    )

    file_name, file_extension = os.path.splitext(input_file)

    if file_extension == '.gz':
        file_stream = gzip.open(input_file, 'rt', encoding=data_encoding)
    elif file_extension == '.zst':
        file_stream = zstd.open(input_file, 'rt', encoding=data_encoding)
    else:
        file_stream = open(input_file, encoding=data_encoding)

    file_stream.readline()

    local_pbar.set_description('Starting ranking computation')
    for line in file_stream:
        line_counter += 1
        local_pbar.update(1)

        if line_counter % args.subsampling != 0:
            continue

        parsed_line = generic_line_parser(
            line, delimiter, args, fw_col_mapping, column_descriptions,
        )

        if len(parsed_line) == len(column_descriptions):
            line_tmp_storage.append(parsed_line)

        else:
            invalid_line_queue.appendleft(str(parsed_line))
            invalid_lines += 1

        # Batches need to be processed on-the-fly
        if len(line_tmp_storage) >= args.minibatch_size:

            importances_batch, bounds_storage, coverage_storage, memory_storage = compute_batch_ranking(
                line_tmp_storage,
                numeric_column_types,
                args,
                cpu_pool,
                column_descriptions,
                logger,
                local_pbar,
            )

            bounds_storage_batch.append(bounds_storage)
            memory_storage_batch.append(memory_storage)
            for k, v in coverage_storage.items():
                local_coverage_object[k].append(v)

            del coverage_storage

            line_tmp_storage = []
            step_timing_checkpoints.append(importances_batch.step_times)
            importances_df += importances_batch.triplet_scores

            if args.heuristic != 'Constant':
                local_pbar.set_description('Creating checkpoint')
                checkpoint_importances_df(importances_df)

    file_stream.close()

    local_pbar.set_description('Parsing the remainder')
    if invalid_lines > 0:
        logger.info(
            f"Detected {invalid_lines} invalid lines. If this number is very high, it's possible your header is off - re-check your data/attribute-feature mappings please!",
        )

        invalid_lines_log = '\n INVALID_LINE ====> '.join(
            list(invalid_line_queue)[0:5],
        )
        logger.info(
            f'5 samples of invalid lines are printed below\n {invalid_lines_log}',
        )

    remaining_batch_size = len(line_tmp_storage)

    if remaining_batch_size > 2**10:
        line_tmp_storage = line_tmp_storage[: args.minibatch_size]
        importances_batch, bounds_storage, coverage_storage, _ = compute_batch_ranking(
            line_tmp_storage,
            numeric_column_types,
            args,
            cpu_pool,
            column_descriptions,
            logger,
            local_pbar,
        )

        for k, v in coverage_storage.items():
            local_coverage_object[k].append(v)

        step_timing_checkpoints.append(importances_batch.step_times)
        importances_df += importances_batch.triplet_scores
        bounds_storage = dict()
        bounds_storage_batch.append(bounds_storage)
        checkpoint_importances_df(importances_df)

    local_pbar.set_description('Wrapping up')
    local_pbar.close()

    return (
        step_timing_checkpoints,
        get_grouped_df(importances_df),
        GLOBAL_CARDINALITY_STORAGE.copy(),
        bounds_storage_batch,
        memory_storage_batch,
        local_coverage_object,
        GLOBAL_RARE_VALUE_STORAGE.copy(),
        GLOBAL_PRIOR_COMB_COUNTS.copy(),
        GLOBAL_COUNTS_STORAGE.copy(),
    )
