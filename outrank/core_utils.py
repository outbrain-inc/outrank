from __future__ import annotations

import csv
import glob
import json
import logging
import os
from collections import Counter
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import xxhash

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

pro_tips = [
    'OutRank can construct subfeatures based on subspaces. Use: --subfeature_mapping "feature_a->feature_b;feature_c<->feature_d;feature_c<->feature_e".',
    'Heuristic MI-numba-randomized offers a great balance of speed and performance.',
    'Heuristic surrogate-lr performs internal cross-validation; keep that in mind.',
    'Consider running OutRank on a smaller sample first; it might be sufficient (--subsampling = a lot).',
    'OutRank supports two types of combinations: unsupervised pairwise ranking (--target_ranking_only=False) and supervised combinations (--interaction_order > 1).',
    'The visualization includes clustering, which can be very insightful!',
    'By default, OutRank includes feature cardinality and coverage in feature names (e.g., card; cov).',
    'Intermediary checkpoints (tmp_checkpoint.tsv) can provide insights during longer runs.',
    'You can rank redundancies of combined features using --interaction_order and --target_ranking_only=False.',
    'Use as many threads as possible for better performance (--num_threads).',
    'Speed up ranking by reducing the feature buffer size (--combination_number_upper_bound) and using --subsampling together.',
    'Not sure which feature transformations to choose? Use --transformers=default for a solid baseline (includes common DS transformations).',
    'Your target can be any feature, allowing you to explain one feature with others.',
    'OutRank uses HyperLogLog for cardinality estimation, useful for understanding cardinalities across datasets.',
    'Each feature is named as featureName(cardinality, coverage in percents) in the final files.',
    'Generate candidate feature transformation ranges using --task=feature_summary_transformers.',
]


def write_json_dump_to_file(args: Any, config_name: str) -> None:

    out_content = json.dumps(args.__dict__)
    with open(config_name, 'w') as out_config:
        out_config.write(out_content)


def internal_hash(input_obj: str) -> str:
    """A generic internal hash used throughout ranking procedure - let's hardcode seed here for sure"""
    return xxhash.xxh32(input_obj, seed=20141025).hexdigest()


@dataclass
class DatasetInformationStorage:
    """A generic class for holding properties of a given type of dataset"""

    data_path: str
    column_names: list[str]
    column_types: set[str]
    col_delimiter: str | None
    encoding: str
    fw_map: dict[str, str] | None


@dataclass
class NumericFeatureSummary:
    """A generic class storing numeric feature statistics"""

    feature_name: str
    minimum: float
    maximum: float
    median: float
    num_unique: int


@dataclass
class NominalFeatureSummary:
    """A generic class storing numeric feature statistics"""

    feature_name: str
    num_unique: int


@dataclass
class BatchRankingSummary:
    """A generic class representing batched ranking results"""

    triplet_scores: list[tuple[str, str, float]]
    step_times: dict[str, Any]


def display_random_tip() -> None:
    TIP_CONTENT = np.random.choice(pro_tips)
    tip_core = f"""
=====>
Random tip: {TIP_CONTENT}
=====>
    """

    print(tip_core)


def get_dataset_info(args: Any):
    if args.data_source == 'ob-raw-dump':
        dataset_info = parse_ob_raw_feature_information(args.data_path)

    elif args.data_source == 'ob-vw':
        dataset_info = parse_ob_vw_feature_information(args.data_path)

    elif args.data_source == 'ob-csv':
        dataset_info = parse_csv_with_description_information(args.data_path)

    elif args.data_source == 'csv-raw':
        dataset_info = parse_csv_raw(args.data_path)
    else:
        raise NotImplementedError(
            'Plase, select a supported data source. Possible sources: {csv-raw, ob-vw, ob-csv}',
        )

    return dataset_info


def display_tool_name() -> None:
    tool_name = """


                        *///////////////.
                     //////////////////////*
                   */////////////////////////.
                  ////////////// */////////////
                  /////////*          /////////
                 //////   /////   ////,   /////
                  ////////     ///    /////////
                  /////   /////  ./////   ////*
                   ,////                 ////
                     *////             ////.
                         ///////*///////


    ░█████╗░██╗░░░██╗████████╗██████╗░░█████╗░███╗░░██╗██╗░░██╗
    ██╔══██╗██║░░░██║╚══██╔══╝██╔══██╗██╔══██╗████╗░██║██║░██╔╝
    ██║░░██║██║░░░██║░░░██║░░░██████╔╝███████║██╔██╗██║█████═╝░
    ██║░░██║██║░░░██║░░░██║░░░██╔══██╗██╔══██║██║╚████║██╔═██╗░
    ╚█████╔╝╚██████╔╝░░░██║░░░██║░░██║██║░░██║██║░╚███║██║░╚██╗
    ░╚════╝░░╚═════╝░░░░╚═╝░░░╚═╝░░╚═╝╚═╝░░╚═╝╚═╝░░╚══╝╚═╝░░╚═╝


    """

    print(tool_name)


def parse_ob_line(
    line_string: str, delimiter: str = '\t', args: Any = None,
) -> list[str]:
    """Outbrain line parsing - generic TSVs"""

    line_string = line_string.strip()
    parts = line_string.split(delimiter)
    return parts


def parse_ob_line_vw(
    line_string: str,
    delimiter: str,
    args: Any = None,
    fw_col_mapping  = None,
    table_header  = None,
    include_namespace_info = False,
) -> list[str | None]:
    """Parse a sparse vw line into a pandas df with pre-defined namespace"""

    all_line_parts = line_string.strip().split('|')
    label_part = all_line_parts[0].split(' ')[0]
    remainder = all_line_parts[1:]
    label = label_part
    remainder_hash = dict()

    # Hash multi-value tuples and store name-val mappings
    for remaining_part in remainder:
        core_parts = remaining_part.strip().split(' ')
        namespace_part = core_parts[0]
        other_parts = '-'.join(x for x in core_parts[1:] if x != '')

        if namespace_part in fw_col_mapping:
            remainder_hash[fw_col_mapping[namespace_part]] = other_parts
        else:
            logging.error(f"Didn't find namespace {namespace_part}")

    # Construct the consistently-mapped instance based on the remainder mapping
    the_real_instance = [
        remainder_hash.get(
            el, None,
        ) for el in table_header[1:]
    ]
    if not include_namespace_info:
        the_real_instance = [
            x[2:] if x is not None else None for x in the_real_instance
        ]

    parts = [label] + the_real_instance

    return parts


def parse_ob_csv_line(
    line_string: str, delimiter: str = ',', args: Any = None,
) -> list[str]:
    """Data can have commas within JSON field dumps"""

    clx = list(csv.reader([line_string])).pop()
    return clx


def generic_line_parser(
    line_string: str,
    delimiter: str,
    args: Any = None,
    fw_col_mapping: Any = None,
    table_header: Any = None,
) -> list[Any]:
    """A generic method aimed to parse data from different sources."""

    if args.data_source == 'ob-raw-dump':
        return parse_ob_line(line_string, delimiter, args)

    elif args.data_source == 'ob-vw':
        return parse_ob_line_vw(
            line_string, delimiter, args, fw_col_mapping, table_header,
        )

    elif args.data_source == 'ob-csv' or args.data_source == 'csv-raw':
        return parse_ob_csv_line(line_string, delimiter, args)

    else:
        raise NotImplementedError(
            'Please, specify a valid --data_source argument!',
        )


def read_reference_json(json_path) -> dict[str, dict]:
    """A helper method for reading a JSON"""
    with open(json_path) as jp:
        return json.load(jp)


def parse_namespace(namespace_path: str) -> tuple[set[str], dict[str, str]]:
    """Parse the feature namespace for type awareness"""

    float_set = set()
    id_feature_map = {}

    with open(namespace_path) as nm:
        for line in nm:
            try:
                namespace_parts = line.strip().split(',')
                if len(namespace_parts) == 2 and '_' not in namespace_parts[0]:
                    fw_id, feature = namespace_parts
                    type_name = 'generic'

                else:
                    fw_id, feature, type_name = namespace_parts

                id_feature_map[fw_id] = feature
                if type_name == 'f32':
                    float_set.add(feature)
            except Exception:
                pass

    return float_set, id_feature_map


def read_column_names(mapping_file: str) -> list[str]:
    """Read the col. header"""

    with open(mapping_file, encoding='utf-8') as mf:
        columns = mf.read().strip().split('\t')
    return columns


def identify_data_file_type(data_path):

    all_files  = set(list(glob.glob(os.path.join(data_path, '*'))))
    gz_pname, zst_pname = 'data.vw.gz', 'data.vw.zst'
    if gz_pname in ''.join(all_files):
        return os.path.join(data_path, gz_pname)
    elif zst_pname in ''.join(all_files):
        return os.path.join(data_path, zst_pname)
    else:
        raise NotImplementedError('Please provide a valid data type .. (gz, zst)')



def parse_ob_vw_feature_information(data_path) -> DatasetInformationStorage:
    """A generic parser of ob-based data"""

    # Get column names
    column_descriptions = os.path.join(data_path, 'vw_namespace_map.csv')
    column_types, fw_map = parse_namespace(column_descriptions)

    # We establish column order here
    column_names = ['label'] + list(fw_map.values())

    data_path = identify_data_file_type(data_path)
    col_delimiter = None
    encoding = 'utf-8'

    return DatasetInformationStorage(
        data_path, column_names, column_types, col_delimiter, encoding, fw_map,
    )


def parse_ob_raw_feature_information(data_path) -> DatasetInformationStorage:
    """A generic parser of ob-based data"""

    # Get column names
    column_types: list[str] = []

    # Get set of numeric columns
    table_header_path = os.path.join(data_path, 'raw_data/0_header/header.csv')
    table_header = read_column_names(table_header_path)

    data_path_train = os.path.join(data_path, 'raw_data/1_train/*')
    col_delimiter = '\t'
    encoding = 'utf-8'

    final_df = []
    core_data_folders = glob.glob(data_path_train)
    for actual_data in core_data_folders:
        for dump in glob.glob(actual_data + '/*'):
            tmp_df = pd.read_csv(
                dump, sep='\t', low_memory=True, dtype='object',
            )
            assert tmp_df.shape[1] == len(table_header)
            tmp_df.columns = table_header
            final_df.append(tmp_df)

    final_df_concat = pd.concat(final_df, axis=0)
    final_path = os.path.join(data_path, 'raw_dump.tsv')
    logging.info(
        f'Stored data dump of dimension {final_df_concat.shape} to {final_path}',
    )
    final_df_concat.to_csv(final_path, sep='\t', index=False)
    data_path = os.path.join(data_path, 'raw_dump.tsv')

    return DatasetInformationStorage(
        data_path, table_header, set(column_types), col_delimiter, encoding, None,
    )


def parse_ob_feature_information(data_path) -> DatasetInformationStorage:
    """A generic parser of ob-based data"""

    # Get column names
    column_names = os.path.join(data_path, 'vw_namespace_map.csv')
    column_types, _ = parse_namespace(column_names)

    # Get set of numeric columns
    table_header_path = os.path.join(data_path, 'raw_data/0_header/header.csv')
    table_header = read_column_names(table_header_path)

    data_path = os.path.join(data_path, 'raw_data/1_train/*')
    col_delimiter = '\t'
    encoding = 'utf-8'

    return DatasetInformationStorage(
        data_path, table_header, column_types, col_delimiter, encoding, None,
    )


def parse_csv_with_description_information(data_path) -> DatasetInformationStorage:
    dataset_description = read_reference_json(
        os.path.join(data_path, 'dataset_desc.json'),
    )
    column_names = []
    column_types = set()
    for feature in dataset_description.get('data_features', []):
        feature_name = feature.get('name')
        column_names.append(feature_name)
        feature_type = feature.get('type', '')
        if 'float' in feature_type or 'Float' in feature_type:
            column_types.add(feature_name)
    col_delimiter = ','
    data_path = os.path.join(data_path, 'data.csv')
    encoding = 'latin1'
    return DatasetInformationStorage(
        data_path, column_names, column_types, col_delimiter, encoding, None,
    )


def parse_csv_raw(data_path) -> DatasetInformationStorage:
    column_types: set[str] = set()

    data_path = os.path.join(data_path, 'data.csv')
    with open(data_path) as inp_data:
        header = inp_data.readline()
    col_delimiter = ','
    column_names = header.strip().split(col_delimiter)
    encoding = 'latin1'
    return DatasetInformationStorage(
        data_path, column_names, column_types, col_delimiter, encoding, None,
    )


def extract_features_from_reference_JSON(json_path: str, combined_features_only = False, all_features = False) -> set[Any]:
    """Given a model's JSON, extract unique features"""

    with open(json_path) as jp:
        content = json.load(jp)

    unique_features = set()
    feature_space = content['desc'].get('features', [])
    if all_features:
        return set(feature_space)

    fields_space = content['desc'].get('fields', [])
    joint_space = feature_space + fields_space

    if combined_features_only:
        return {feature for feature in feature_space if len(feature.split(','))>1}

    for feature_tuple in joint_space:
        for individual_feature in feature_tuple.split(','):
            unique_features.add(individual_feature)

    return unique_features


def summarize_feature_bounds_for_transformers(
    bounds_object_storage: Any,
    feature_types: list[str],
    task_name: str,
    label_name: str,
    granularity: int = 15,
    output_summary_table_only: bool = False,
):
    """summarization auxilliary method for generating JSON-based specs"""

    if bounds_object_storage is None:
        logging.info('Bounds storage object is empty.')
        exit()

    final_storage = defaultdict(list)
    for el in bounds_object_storage:
        if isinstance(el, dict):
            for k, v in el.items():
                final_storage[k].append(v)

    summary_table_rows = []
    for k, v in final_storage.items():
        # Conduct local aggregation + bound changes
        if k in feature_types and k != label_name:
            minima, maxima, medians, uniques = [], [], [], []
            for feature_summary in v:
                minima.append(feature_summary.minimum)
                maxima.append(feature_summary.maximum)
                medians.append(feature_summary.median)
                uniques.append(feature_summary.num_unique)
            summary_table_rows.append(
                [
                    k,
                    round(np.min(minima), 2),
                    round(np.max(maxima), 2),
                    round(np.median(medians), 2),
                    int(np.mean(uniques)),
                ],
            )

    if len(summary_table_rows) == 0:
        logging.info('No numeric features to summarize.')
        return None

    summary_table: pd.Dataframe = pd.DataFrame(summary_table_rows)
    summary_table.columns = [
        'Feature',
        'Minimum',
        'Maximum',
        'Median',
        'Num avg. unique (batch)',
    ]

    if output_summary_table_only:
        return summary_table

    if len(summary_table) == 0:
        logging.info('Summary table empty, skipping transformer generation ..')
        return

    if task_name == 'feature_summary_transformers':
        transformers_per_feature = defaultdict(list)

        # Take care of weights first -> range is pre-defined
        for k, v in final_storage.items():
            if label_name in k or 'dummy' in k:
                continue

            weight_template = {
                'feature': k,
                'src_features': [k],
                'transformations': ['Weight'],
                'weights': [0, 0.5, 1.5, 2, 3, 10],
            }
            transformers_per_feature[k].append(weight_template)

        # Consider numeric transformations - pairs and single ones
        for enx, row in summary_table.iterrows():
            if row.Feature == 'dummy':
                continue
            try:
                actual_range = (
                    np.arange(
                        row['Minimum'],
                        row['Maximum'],
                        (row['Maximum'] - row['Minimum']) / granularity,
                    )
                    .round(2)
                    .tolist()
                )
                binner_template = {
                    'feature': f'{row.Feature}',
                    'src_features': [row.Feature],
                    'transformations': [
                        'BinnerSqrt',
                        'BinnerLog',
                        'BinnerSqrtPlain',
                        'BinnerLogPlain',
                    ],
                    'n': actual_range,
                    'resolutions': [0.1, 2, 4, 8, 16, 32, 64, 128],
                }

            except Exception as es:
                logging.info(
                    f'\U0001F631 Encountered {es}. The problematic feature is: {row}, skipping transformer for this feature ..',
                )

            transformers_per_feature[row.Feature].append(binner_template)

            # We want the full loop here, due to asymmetry of transformation(s)
            for enx_second, row_second in summary_table.iterrows():
                if enx_second < enx:
                    continue

                # The n values are defined based on maxima of the second feature
                if row_second.Feature != row.Feature:
                    n_bound = round(row_second['Median'] + row['Median'], 2)
                    max_bound = round(
                        min(row_second['Maximum'], row['Maximum']), 2,
                    )
                    min_bound = round(
                        row_second['Minimum'] + row['Minimum'], 2,
                    )
                    range_spectrum = sorted(
                        list(
                            {
                                0.0,
                                min_bound,
                                n_bound / 10,
                                n_bound / 5,
                                n_bound,
                                max_bound,
                            },
                        ),
                    )

                    range_spectrum = [x for x in range_spectrum if x >= 0]
                    binner_pair_template = {
                        'feature': f'{row.Feature}Ratio{row_second.Feature}',
                        'src_features': [row.Feature, row_second.Feature],
                        'transformations': ['BinnerLogRatioPlain', 'BinnerLogRatio'],
                        'n': range_spectrum,
                        'resolutions': [0.1, 2, 4, 8, 16, 32, 64, 128],
                    }

                    binner_pair_template_second = {
                        'feature': f'{row_second.Feature}Ratio{row.Feature}',
                        'src_features': [row_second.Feature, row.Feature],
                        'transformations': ['BinnerLogRatioPlain', 'BinnerLogRatio'],
                        'n': range_spectrum,
                        'resolutions': [0.1, 2, 4, 8, 16, 32, 64, 128],
                    }

                    transformers_per_feature[row.Feature].append(
                        binner_pair_template,
                    )
                    transformers_per_feature[row.Feature].append(
                        binner_pair_template_second,
                    )

        binner_templates = []
        for k, v in transformers_per_feature.items():
            for transformer_struct in v:
                binner_templates.append(transformer_struct)

        logging.info(
            f'Generated {len(binner_templates)} transformation search specifications.\n',
        )
        namespace_full = f'"random_grid_feature_transform": {json.dumps(binner_templates)}, "random_grid_epochs": 512'
        logging.info('Generated transformations below:\n')
        print(namespace_full)


def summarize_rare_counts(
    term_counter: Any,
    args: Any,
    cardinality_object: Any,
    object_info: DatasetInformationStorage,
) -> None:
    """Write rare values"""

    out_df_rows = []
    logging.info(
        f'Rare value summary (freq <= {args.rare_value_count_upper_bound}) follows ..',
    )

    for namespace_tuple, count in term_counter.items():
        namespace, value = namespace_tuple
        out_df_rows.append([namespace, value, count])
    out_df: pd.DataFrame = pd.DataFrame(out_df_rows)
    out_df.columns = ['Namespace', 'value', 'Count']
    out_df.to_csv(
        os.path.join(args.output_folder, 'rare_values.tsv'), sep='\t', index=False,
    )
    logging.info(f'Wrote rare values to {args.output_folder}/rare_values.tsv')

    overall_rare_counts = Counter(out_df.Namespace.values)
    sorted_counts = sorted(
        overall_rare_counts.items(), key=lambda pair: pair[1], reverse=True,
    )
    for k, v in sorted_counts:
        logging.info(f'Namespace: {k} ---- Rare values observed: {v}')

    final_df_rows = []
    for k, v in sorted_counts:
        cardinality = len(cardinality_object[k])
        rare_proportion = np.round(100 * (v / cardinality), 2)
        col_type = 'nominal'
        if k in object_info.column_types:
            col_type = 'numeric'
        final_df_rows.append(
            {
                'rare_proportion': rare_proportion,
                'feature_type': col_type,
                'feature_name': k,
            },
        )

    final_df: pd.DataFrame = pd.DataFrame(final_df_rows)
    final_df = final_df.sort_values(by=['rare_proportion'])
    logging.info(
        f'Wrote feature sparsity summary to {args.output_folder}/feature_sparsity_summary.tsv',
    )
    final_df.to_csv(
        f'{args.output_folder}/feature_sparsity_summary.tsv', index=False, sep='\t',
    )


def is_prior_heuristic(args: Any) -> bool:
    if args.heuristic in {'surrogate-SGD', 'surrogate-SVM', 'surrogate-SGD-RP'} and args.reference_model_JSON:
        return True
    return False


def get_num_of_instances(fname: str) -> int:
    """Count the number of lines in a file, fast - useful for progress logging"""

    def _make_gen(reader):
        while True:
            b = reader(2**16)
            if not b:
                break
            yield b

    with open(fname, 'rb') as f:
        count = sum(buf.count(b'\n') for buf in _make_gen(f.raw.read))
    return count
