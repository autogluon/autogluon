"""
Default constants and utility functions for working with dictionaries that represent mappings from
model metadata in autogluon.forecasting to column names in the data sets provided.
"""
from typing import Any, Dict, List, Union

DEFAULT_TARGET_COLUMN_NAME = "target"
MetadataDict = Dict[str, Union[str, List[str]]]


def get_prototype_metadata_dict() -> MetadataDict:
    return {
        "target": DEFAULT_TARGET_COLUMN_NAME,  # column name of the target time series to be forecasted
        "known_feature_real": [],  # real valued features known in the past and future
        "known_feature_cat": [],  # categorical features known in the past and future
        "observed_feature_real": [],  # real valued features known in the past alone
        "observed_feature_cat": [],  # categorical features known in the past alone
    }


def infer_metadata(dataset: Any, metadata: MetadataDict) -> MetadataDict:  # noqa
    return get_prototype_metadata_dict()  # TODO: implement
