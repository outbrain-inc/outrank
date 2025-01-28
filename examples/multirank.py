from __future__ import annotations

import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import linkage


def rbo_score(l1, l2, p=0.9):
    """
    Calculate the Rank-Biased Overlap (RBO) score.

    Args:
        l1 (list): Ranked list of elements.
        l2 (list): Ranked list of elements.
        p (float): Persistence probability (0 <= p < 1), default is 0.9

    Returns:
        float: RBO score, a value between 0 and 1.
    """
    if l1 == l2:
        return 1.0

    len1, len2 = len(l1), len(l2)
    if len1 == 0 or len2 == 0:
        return 0.0

    overlap, rbo, depth = 0, 0, 0
    seen = set()

    for i in range(max(len1, len2)):
        if i < len1 and l1[i] not in seen:
            overlap += 1
            seen.add(l1[i])
        if i < len2 and l2[i] not in seen:
            overlap += 1
            seen.add(l2[i])

        depth += 1
        weight = (p ** (depth - 1)) / depth
        rbo += (overlap / depth) * weight

    return rbo * (1 - p)

if __name__ == '__main__':

    # Define the number of top features to consider
    top_n = 10

    # Define different sizes and corresponding folder names
    sizes = [100000, 15000, 20000, 30000, 50000, 70000, 230000, 25000, 35000, 15000]
    input_folders = [f'../examples/df{i+1}' for i in range(10)]
    output_folders = [f'./output_df{i+1}' for i in range(10)]

    # Initialize a DataFrame to accumulate results
    accumulated_results = pd.DataFrame()

    # Loop over the sizes and folders
    for i, (size, input_folder, output_folder) in enumerate(zip(sizes, input_folders, output_folders), start=1):
        # Generate data set
        dataset_id = f'dataset_{i}'  # Identifier for each data set
        print(f'Generating data set for size {size} with id {dataset_id}')
        os.system(f'python ../benchmarks/generator_third_order.py --size {size} --output_df_name {input_folder}')

        # Run ranking
        print(f'Running ranking for data set {input_folder}')
        os.system(f"""
            outrank \
            --task all \
            --data_path {input_folder} \
            --data_source csv-raw \
            --heuristic MI-numba-randomized \
            --target_ranking_only True \
            --combination_number_upper_bound 2048 \
            --num_threads 12 \
            --output_folder {output_folder} \
            --subsampling 1
        """)

        # Read and accumulate the results from 'feature_singles.tsv'
        feature_singles_path = os.path.join(output_folder, 'feature_singles.tsv')
        if os.path.exists(feature_singles_path):
            print(f'Reading results from {feature_singles_path}')
            df_singles = pd.read_csv(feature_singles_path, sep='\t')
            df_singles['size'] = size  # Include the size information in the results
            df_singles['dataset_id'] = dataset_id  # Include the dataset identifier

            # Ensure 'Score' column naming correctness
            score_column = 'Score' if 'Score' in df_singles.columns else 'Score MI-numba-randomized'

            # Include rank based on Score
            df_singles['rank'] = df_singles[score_column].rank(ascending=False)

            # Clean the Feature names by taking only the part before the "-"
            df_singles['Feature-clean'] = df_singles['Feature'].apply(lambda x: x.split('-')[0])

            # Accumulate the results
            accumulated_results = pd.concat([accumulated_results, df_singles], ignore_index=True)
        else:
            print(f'Warning: {feature_singles_path} does not exist!')

        # Data cleanup
        print(f'Cleaning up data set {input_folder} and output {output_folder}')
        if os.path.exists(input_folder):
            shutil.rmtree(input_folder)

        if os.path.exists(output_folder):
            shutil.rmtree(output_folder)

    # Compute average and standard deviation of ranks for each feature
    rank_stats = accumulated_results.groupby('Feature-clean').agg(
        avg_rank=('rank', 'mean'),
        std_rank=('rank', 'std'),
    ).reset_index()

    # Save accumulated results and rank statistics
    output_csv_path = './accumulated_feature_singles_results.csv'
    rank_stats_csv_path = './feature_rank_stats.csv'

    print(f'Saving accumulated results to {output_csv_path}')
    accumulated_results.to_csv(output_csv_path, sep='\t', index=False)

    print(f'Saving rank statistics to {rank_stats_csv_path}')
    rank_stats.to_csv(rank_stats_csv_path, sep='\t', index=False)

    # Compute pairwise similarity using RBO for top n features
    datasets = accumulated_results['dataset_id'].unique()
    similarity_matrix = np.zeros((len(datasets), len(datasets)))

    for i, dataset_i in enumerate(datasets):
        for j, dataset_j in enumerate(datasets):
            if i <= j:  # Compute only for upper triangle and diagonal
                ranks_i = accumulated_results[accumulated_results['dataset_id'] == dataset_i].nlargest(top_n, 'rank').set_index('Feature-clean')['rank']
                ranks_j = accumulated_results[accumulated_results['dataset_id'] == dataset_j].nlargest(top_n, 'rank').set_index('Feature-clean')['rank']

                # Align the series
                common_features = ranks_i.index.intersection(ranks_j.index)
                if len(common_features) > 0:
                    ranks_i = ranks_i[common_features]
                    ranks_j = ranks_j[common_features]
                    rbo_similarity = round(rbo_score(ranks_i.tolist(), ranks_j.tolist()), 3)
                    similarity_matrix[i, j] = rbo_similarity
                    similarity_matrix[j, i] = rbo_similarity

    # Convert the similarity matrix to DataFrame for saving
    similarity_df = pd.DataFrame(similarity_matrix, index=datasets, columns=datasets)
    similarity_matrix_path = './dataset_similarity_matrix.tsv'

    print(f'Saving similarity matrix to {similarity_matrix_path}')
    similarity_df.to_csv(similarity_matrix_path, sep='\t')

    # Visualization via dendrogram
    def plot_dendrogram(similarity_matrix, datasets):
        # Convert similarity matrix to distance matrix
        distance_matrix = 1 - similarity_matrix

        # Perform hierarchical/agglomerative clustering
        linkage_matrix = linkage(distance_matrix, 'ward')

        # Plot the dendrogram
        plt.figure(figsize=(10, 7))
        dendrogram(linkage_matrix, labels=datasets, leaf_rotation=90)
        plt.title('Dendrogram of Dataset Similarities')
        plt.xlabel('Dataset')
        plt.ylabel('Distance')
        plt.tight_layout()
        plt.savefig('Dendrogram_all.pdf', dpi=300)

    print('Plotting dendrogram...')
    plot_dendrogram(similarity_matrix, datasets)

    print('Loop completed successfully, data has been cleaned up, rank statistics, and similarity matrix have been computed.')
