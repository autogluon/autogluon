import logging
import math

from pandas import DataFrame

from ..features.types import get_type_map_raw


logger = logging.getLogger(__name__)


# TODO: Documentation
def get_approximate_df_mem_usage(df: DataFrame, sample_ratio=0.2):
    if sample_ratio >= 1:
        return df.memory_usage(deep=True)
    else:
        num_rows = len(df)
        num_rows_sample = math.ceil(sample_ratio * num_rows)
        sample_ratio = num_rows_sample / num_rows
        exact_dtypes = ['int', 'float']
        category_dtypes = ['category']
        dtypes_raw = get_type_map_raw(df)
        columns_category = [column for column in df if dtypes_raw[column] in category_dtypes]
        columns_inexact = [column for column in df if dtypes_raw[column] not in exact_dtypes + category_dtypes]
        memory_usage = df.memory_usage()
        if columns_category:
            for column in columns_category:
                num_categories = len(df[column].cat.categories)
                num_categories_sample = math.ceil(sample_ratio * num_categories)
                sample_ratio_cat = num_categories_sample / num_categories
                memory_usage[column] = df[column].cat.codes.dtype.itemsize * num_rows + df[column].cat.categories[:num_categories_sample].memory_usage(deep=True) / sample_ratio_cat
        if columns_inexact:
            memory_usage_inexact = df[columns_inexact].head(num_rows_sample).memory_usage(deep=True)[columns_inexact] / sample_ratio
            memory_usage = memory_usage_inexact.combine_first(memory_usage)
        return memory_usage
