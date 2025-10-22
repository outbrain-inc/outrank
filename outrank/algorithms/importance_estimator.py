from __future__ import annotations

import logging
import operator
import traceback
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn import random_projection
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC

from outrank.algorithms.feature_ranking import ranking_cov_alignment, ranking_mi_numba_opt

logger = logging.getLogger('syn-logger')
logger.setLevel(logging.DEBUG)

NUM_FOLDS  = 2
SVD_DIMS = 8

try:
    from outrank.algorithms.feature_ranking import ranking_mi_numba
    numba_available = True
except ImportError:
    traceback.print_exc()
    numba_available = False

try:
    from outrank.algorithms.feature_ranking import ranking_mi_multivalue
    multivalue_available = True
except ImportError:
    traceback.print_exc()
    multivalue_available = False

def sklearn_MI(vector_first: np.ndarray, vector_second: np.ndarray) -> float:
    # Vectors are already shaped correctly by generate_data_for_ranking
    # vector_first is (n, 1) or (n, m), vector_second is (n,)
    if vector_first.ndim == 1:
        vector_first = vector_first.reshape(-1, 1)
    return mutual_info_classif(
        vector_first, vector_second, discrete_features=True,
    )[0]

def sklearn_surrogate(
    vector_first: np.ndarray, vector_second: np.ndarray,  surrogate_model: str,
) -> float:
    # Vectors are already shaped correctly by generate_data_for_ranking
    # vector_first is (n, 1) or (n, m), so no need to reshape
    if vector_first.ndim == 1:
        vector_first = vector_first.reshape(-1, 1)

    X = OneHotEncoder().fit_transform(vector_first)

    if '-SVD' in surrogate_model and X.shape[1] > 2:
        # yes this is not super correct due to embedding full data first, but it's much faster + seems to offer same results anyways.
        X = TruncatedSVD(n_components=min(SVD_DIMS, X.shape[1])).fit_transform(X)

    clf = initialize_classifier(surrogate_model, n_dim=min(X.shape[1], 1024))
    scores = cross_val_score(clf, X, vector_second, scoring='neg_log_loss', cv=NUM_FOLDS)
    return 1 + np.median(scores)

def numba_mi(vector_first: np.ndarray, vector_second: np.ndarray, heuristic: str, mi_stratified_sampling_ratio: float) -> float:
    cardinality_correction = heuristic == 'MI-numba-randomized'

    # Vectors are already shaped correctly by generate_data_for_ranking
    # vector_first is (n, 1) or (n, m), we need to convert to 1D for numba
    if vector_first.ndim == 2:
        if vector_first.shape[1] == 1:
            vector_first = vector_first.reshape(-1)
        else:
            # Multi-column case: aggregate into single column
            vector_first = np.apply_along_axis(lambda x: np.abs(np.max(x) - np.sum(x)), 1, vector_first)

    return ranking_mi_numba.mutual_info_estimator_numba(
        vector_first.astype(np.int32),
        vector_second.astype(np.int32),
        approximation_factor=np.float32(mi_stratified_sampling_ratio),
        cardinality_correction=cardinality_correction,
    )
    
def numba_mi_opt(vector_first: np.ndarray, vector_second: np.ndarray, heuristic: str, mi_stratified_sampling_ratio: float) -> float:

    cardinality_correction = heuristic == 'MI-numba-randomized-opt'

    # Preprocess vector_first to ensure it is a 1D array. This handles cases
    # where features might be multi-column (e.g., one-hot encoded).
    if vector_first.ndim == 2:
        if vector_first.shape[1] > 1:
            vector_first = np.apply_along_axis(lambda x: np.abs(np.max(x) - np.sum(x)), 1, vector_first)
        else:
            vector_first = vector_first.reshape(-1)

    return ranking_mi_numba_opt.mutual_info_estimator_numba_opt(
        vector_first.astype(np.int32),
        vector_second.astype(np.int32),
        approximation_factor=np.float32(mi_stratified_sampling_ratio),
        cardinality_correction=cardinality_correction,
    )

def sklearn_mi_adj(vector_first: np.ndarray, vector_second: np.ndarray) -> float:
    # adjusted_mutual_info_score expects 1D arrays
    v1 = vector_first.reshape(-1) if vector_first.ndim > 1 else vector_first
    return adjusted_mutual_info_score(v1, vector_second)

def multivalue_mi_jaccard(vector_first: np.ndarray, vector_second: np.ndarray) -> float:
    """Compute mutual information between multivalue features using Jaccard similarity."""
    if not multivalue_available:
        logger.warning('Multivalue MI not available, falling back to standard MI')
        return sklearn_MI(vector_first, vector_second)
    
    return ranking_mi_multivalue.multivalue_mutual_info_estimator(
        vector_first, vector_second, algorithm='jaccard'
    )

def multivalue_mi_overlap(vector_first: np.ndarray, vector_second: np.ndarray) -> float:
    """Compute mutual information between multivalue features using overlap-based approach."""
    if not multivalue_available:
        logger.warning('Multivalue MI not available, falling back to standard MI')
        return sklearn_MI(vector_first, vector_second)
    
    return ranking_mi_multivalue.multivalue_mutual_info_estimator(
        vector_first, vector_second, algorithm='overlap'
    )

def multivalue_mi_set_based(vector_first: np.ndarray, vector_second: np.ndarray, cardinality_correction: bool = False) -> float:
    """Compute mutual information between multivalue features using set-based approach.
    
    Args:
        vector_first: First multivalue feature vector
        vector_second: Second multivalue feature vector
        cardinality_correction: If True, apply cardinality correction to prevent inflation
    """ 
    if not multivalue_available:
        logger.warning('Multivalue MI not available, falling back to standard MI')
        return sklearn_MI(vector_first, vector_second)
    
    return ranking_mi_multivalue.multivalue_mutual_info_estimator(
        vector_first, vector_second, algorithm='set_based', cardinality_correction=cardinality_correction
    )

def generate_data_for_ranking(combination: tuple[str, str], reference_model_features: list[str], args: Any, tmp_df: pd.DataFrame) -> tuple(np.ndarray, np.ndrray):
    feature_one, feature_two = combination

    if feature_one == args.label_column:
        feature_one = feature_two
        feature_two = args.label_column

    if args.reference_model_JSON:
        vector_first = tmp_df[list(reference_model_features) + [feature_one]].values
    else:
        vector_first = tmp_df[feature_one].values

    vector_second = tmp_df[feature_two].values
    
    # Ensure vectors have consistent shape to avoid repeated reshaping downstream
    # vector_first can be 1D or 2D (multi-column), vector_second is always 1D
    if vector_first.ndim == 1:
        vector_first = vector_first.reshape(-1, 1)
    if vector_second.ndim != 1:
        vector_second = vector_second.reshape(-1)
    
    return vector_first, vector_second


def conduct_feature_ranking(vector_first: np.ndarray, vector_second: np.ndarray, args: Any) -> float:

    heuristic = args.heuristic
    score = 0.0

    if heuristic == 'MI':
        score = sklearn_MI(vector_first, vector_second)

    elif heuristic in {'surrogate-SGD', 'surrogate-SVM', 'surrogate-SGD-RP', 'surrogate-SGD-SVD'}:
        score = sklearn_surrogate(vector_first, vector_second, heuristic)

    elif heuristic == 'max-value-coverage':
        score = ranking_cov_alignment.max_pair_coverage(vector_first, vector_second)

    elif heuristic == 'MI-numba-randomized':
        score = numba_mi(vector_first, vector_second, heuristic, args.mi_stratified_sampling_ratio)
    
    elif heuristic == 'MI-numba-randomized-opt':
        score = numba_mi_opt(vector_first, vector_second, heuristic, args.mi_stratified_sampling_ratio)

    elif heuristic == 'AMI':
        score = sklearn_mi_adj(vector_first, vector_second)

    elif heuristic == 'MI-multivalue-jaccard':
        score = multivalue_mi_jaccard(vector_first, vector_second)

    elif heuristic == 'MI-multivalue-overlap':
        score = multivalue_mi_overlap(vector_first, vector_second)

    elif heuristic == 'MI-multivalue-set':
        score = multivalue_mi_set_based(vector_first, vector_second)
    
    elif heuristic == 'MI-multivalue-set-randomized':
        score = multivalue_mi_set_based(vector_first, vector_second, cardinality_correction=True)

    elif heuristic == 'correlation-Pearson':
        # pearsonr expects 1D arrays
        v1 = vector_first.reshape(-1) if vector_first.ndim > 1 else vector_first
        score = pearsonr(v1, vector_second)[0]

    elif heuristic == 'Constant':
        score = 0.0

    else:
        logger.warning(f'{heuristic} not defined!')
        score = 0.0

    return score

def get_importances_estimate_pairwise(combination: tuple[str, str], reference_model_features: list[str], args: Any, tmp_df: pd.DataFrame) -> tuple[str, str, float]:

    feature_one, feature_two = combination
    inputs_encoded, output_encoded = generate_data_for_ranking(combination, reference_model_features, args, tmp_df)

    ranking_score = conduct_feature_ranking(inputs_encoded, output_encoded, args)

    return feature_one, feature_two, ranking_score


def rank_features_3MR(
    relevance_dict: dict[str, float],
    redundancy_dict: dict[tuple[Any, Any], Any],
    relational_dict: dict[tuple[Any, Any], Any],
    strategy: str = 'median',
    alpha: float = 1.0,
    beta: float = 1.0,
) -> pd.DataFrame:
    all_features = set(relevance_dict.keys())
    most_important_feature = max(relevance_dict.items(), key=operator.itemgetter(1))[0]
    ranked_features = [most_important_feature]

    def calc_higher_order(feature: str, is_redundancy: bool = True) -> float:
        values = []
        for feat in ranked_features:
            interaction_tuple = (feat, feature)
            if is_redundancy:
                values.append(redundancy_dict.get(interaction_tuple, 0))
            else:
                values.append(relational_dict.get(interaction_tuple, 0))
        return np.median(values) if strategy == 'median' else (np.mean(values) if strategy == 'mean' else sum(values))

    while len(ranked_features) < len(all_features):
        top_importance = -np.inf
        most_important_feature = None

        for feat in all_features - set(ranked_features):
            feature_redundancy = calc_higher_order(feat)
            feature_relation = calc_higher_order(feat, False)
            feature_relevance = relevance_dict[feat]
            importance = feature_relevance - alpha * feature_redundancy + beta * feature_relation

            if importance > top_importance:
                top_importance = importance
                most_important_feature = feat

        ranked_features.append(most_important_feature)

    return pd.DataFrame({'Feature': ranked_features, '3MR_Ranking': range(1, len(ranked_features) + 1)})

def get_importances_estimate_nonmyopic(args: Any, tmp_df: pd.DataFrame):
    pass

def initialize_classifier(surrogate_model: str, n_dim: int) -> Any:

    if 'surrogate-LR' in surrogate_model:
        return LogisticRegression(max_iter=100000)

    elif 'surrogate-SVM' in surrogate_model:
        return SVC(gamma='auto', probability=True)

    elif 'surrogate-SGD-RP' in surrogate_model:
        clf = Pipeline([('proj', random_projection.SparseRandomProjection(n_components=n_dim)), ('reg', SGDClassifier(max_iter=100000, loss='log_loss'))])
        return clf

    elif 'surrogate-SGD' in surrogate_model:
        return SGDClassifier(max_iter=100000, loss='log_loss')

    else:
        logger.warning(f'The chosen surrogate model {surrogate_model} is not supported, falling back to surrogate-SGD')
        return SGDClassifier(max_iter=100000, loss='log_loss')
