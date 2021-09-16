import logging

import numpy as np
import pandas as pd
from pandas import DataFrame, IntervalIndex, Series

logger = logging.getLogger(__name__)


def bin_column(series: Series, bins, dtype):
    return np.digitize(series, bins=bins, right=True).astype(dtype)


# TODO: Rewrite with normalized value counts as binning technique, will be more performant and optimal
def generate_bins(X_features: DataFrame, features_to_bin: list, ideal_bins: int = 10):
    X_len = len(X_features)
    starting_cats = 1000
    bin_index_starting = [np.floor(X_len * (num + 1) / starting_cats) for num in range(starting_cats - 1)]
    bin_epsilon = 0.000000001
    bin_mapping = dict()
    max_iterations = 20
    for column in features_to_bin:
        num_cats_initial = starting_cats
        bins_value_counts = X_features[column].value_counts(ascending=False, normalize=True)
        max_bins = len(bins_value_counts)

        if max_bins <= ideal_bins:
            bins = pd.Series(data=sorted(X_features[column].unique()))
            num_cats_initial = max_bins
            cur_len = max_bins
            bin_index = list(range(num_cats_initial))
            interval_index = get_bins(bins=bins, bin_index=bin_index, bin_epsilon=bin_epsilon)
        else:
            cur_len = X_len
            bins = X_features[column].sort_values(ascending=True)
            interval_index = get_bins(bins=bins, bin_index=bin_index_starting, bin_epsilon=bin_epsilon)

        # TODO: max_desired_bins and min_desired_bins are currently equivalent, but in future they will be parameterized to allow for flexibility.
        max_desired_bins = min(ideal_bins, max_bins)
        min_desired_bins = min(ideal_bins, max_bins)

        is_satisfied = (len(interval_index) >= min_desired_bins) and (len(interval_index) <= max_desired_bins)

        num_cats_current = num_cats_initial
        cur_iteration = 0
        while not is_satisfied:
            ratio_reduction = max_desired_bins / len(interval_index)
            num_cats_current = int(np.floor(num_cats_current * ratio_reduction))
            bin_index = [np.floor(cur_len * (num + 1) / num_cats_current) for num in range(num_cats_current - 1)]
            interval_index = get_bins(bins=bins, bin_index=bin_index, bin_epsilon=bin_epsilon)

            if (len(interval_index) >= min_desired_bins) and (len(interval_index) <= max_desired_bins):
                is_satisfied = True
                # print('satisfied', column, len(interval_index))
            cur_iteration += 1
            if cur_iteration >= max_iterations:
                is_satisfied = True
                # print('max_iterations met, stopping prior to satisfaction!', column, len(interval_index))

        bins_final = interval_index.right.values
        bin_mapping[column] = bins_final
    return bin_mapping


# TODO: Clean code
# TODO: Consider re-using bins variable instead of making bins_2-7 variables
# bins is a sorted int/float series, ascending=True
def get_bins(bins: Series, bin_index: list, bin_epsilon: float) -> IntervalIndex:
    max_val = bins.max()
    bins_2 = bins.iloc[bin_index]
    bins_3 = list(bins_2.values)
    bins_unique = sorted(list(set(bins_3)))
    bins_with_epsilon_max = set([i for i in bins_unique] + [i - bin_epsilon for i in bins_unique if i == max_val])
    removal_bins = set([bins_unique[index - 1] for index, i in enumerate(bins_unique[1:], start=1) if i == max_val])
    bins_4 = sorted(list(bins_with_epsilon_max - removal_bins))
    bins_5 = [np.inf if (x == max_val) else x for x in bins_4]
    bins_6 = sorted(list(set([-np.inf] + bins_5 + [np.inf])))
    bins_7 = [(bins_6[i], bins_6[i + 1]) for i in range(len(bins_6) - 1)]
    interval_index = IntervalIndex.from_tuples(bins_7)
    return interval_index
