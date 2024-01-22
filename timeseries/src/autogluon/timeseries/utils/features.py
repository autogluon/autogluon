import logging
import reprlib
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from autogluon.common.features.feature_metadata import FeatureMetadata

import numpy as np
import pandas as pd

from autogluon.common.features.types import R_FLOAT, R_INT
from autogluon.features.generators import (
    AsTypeFeatureGenerator,
    CategoryFeatureGenerator,
    IdentityFeatureGenerator,
    PipelineFeatureGenerator,
)
from autogluon.timeseries import TimeSeriesDataFrame
from pandas import DataFrame

logger = logging.getLogger(__name__)


@dataclass
class CovariateMetadata:
    """Provides mapping from different covariate types to columns in the dataset."""

    static_features_cat: List[str] = field(default_factory=list)
    static_features_real: List[str] = field(default_factory=list)
    known_covariates_real: List[str] = field(default_factory=list)
    known_covariates_cat: List[str] = field(default_factory=list)
    past_covariates_real: List[str] = field(default_factory=list)
    past_covariates_cat: List[str] = field(default_factory=list)


class ContinuousAndCategoricalFeatureGenerator(PipelineFeatureGenerator):
    """Generates categorical and continuous features for time series models."""

    def __init__(self, verbosity: int = 0, **kwargs):
        generators = [
            CategoryFeatureGenerator(minimum_cat_count=1, fillna="mode"),
            IdentityFeatureGenerator(infer_features_in_args={"valid_raw_types": [R_INT, R_FLOAT]}),
        ]
        super().__init__(
            generators=[generators],
            post_generators=[],
            pre_generators=[AsTypeFeatureGenerator(convert_bool=False)],
            pre_enforce_types=False,
            pre_drop_useless=False,
            verbosity=verbosity,
            **kwargs,
        )

    def fit_transform(self, X: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        if isinstance(X, TimeSeriesDataFrame):
            X = pd.DataFrame(X)
        return super().fit_transform(X, *args, **kwargs)


class TimeSeriesFeatureGenerator:
    """Takes care of preprocessing for static_features and past/known covariates.

    Covariates are all converted to float dtype. Static features, if present, are all converted to categorical & float
    dtypes.
    """

    def __init__(self, target: str, known_covariates_names: List[str]):
        self.target = target
        self._is_fit = False
        self.known_covariates_names = list(known_covariates_names)
        self.past_covariates_names = []
        self.known_covariates_pipeline = ContinuousAndCategoricalFeatureGenerator()
        self.past_covariates_pipeline = ContinuousAndCategoricalFeatureGenerator()
        self.static_feature_pipeline = ContinuousAndCategoricalFeatureGenerator()
        self.covariate_metadata: CovariateMetadata = None

    @property
    def required_column_names(self) -> List[str]:
        return [self.target] + list(self.known_covariates_names) + list(self.past_covariates_names)

    @staticmethod
    def _detect_inferred_types(
        transformed_df: pd.DataFrame, original_column_names: List[str]
    ) -> Tuple[List[str], List[str]]:
        """Return names of categorical and real-valued columns, and log the inferred column types."""
        cat_column_names = []
        real_column_names = []
        unused_columns = transformed_df.columns.difference(original_column_names)
        for column_name, column_dtype in transformed_df.dtypes.items():
            if pd.api.types.is_categorical_dtype(column_dtype):
                cat_column_names.append(column_name)
            elif pd.api.types.is_numeric_dtype(column_dtype):
                real_column_names.append(column_name)
            else:
                unused_columns.append(column_name)

        logger.info("Inferred types of {name}:")
        logger.info(f"\tcategorical:        {cat_column_names}")
        logger.info(f"\tcontinuous (float): {real_column_names}")
        if len(unused_columns) > 0:
            logger.info(f"\t\tremoved (uninformative columns): {unused_columns}")
        return cat_column_names, real_column_names

    @staticmethod
    def _convert_numerical_features_to_float(df: pd.DataFrame, float_dtype=np.float64) -> pd.DataFrame:
        """In-place convert the dtype of all numerical (float or int) columns to the given float dtype."""
        numeric_columns = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
        df[numeric_columns] = df[numeric_columns].astype(float_dtype)
        return df

    def fit(self, data: TimeSeriesDataFrame) -> None:
        assert not self._is_fit, f"{self.__class__.__name__} has already been fit"

        self.past_covariates_names = []
        for column in data.columns:
            if column != self.target and column not in self.known_covariates_names:
                self.past_covariates_names.append(column)

        logger.info("\nProvided dataset contains following columns:")
        logger.info(f"\ntarget:           '{self.target}'")
        if len(self.known_covariates_names) > 0:
            logger.info(f"\nknown covariates: {self.known_covariates_names}")
            known_covariates_df = self.known_covariates_pipeline.fit_transform(data[self.known_covariates_names])
            known_covariates_cat, known_covariates_real = self._detect_inferred_types(
                known_covariates_df, original_column_names=self.known_covariates_names
            )
        else:
            known_covariates_cat = []
            known_covariates_real = []

        if len(self.past_covariates_names) > 0:
            logger.info(f"\npast covariates:  {self.past_covariates_names}")
            past_covariates_df = self.past_covariates_pipeline.fit_transform(data[self.past_covariates_names])
            past_covariates_cat, past_covariates_real = self._detect_inferred_types(
                past_covariates_df, original_column_names=self.past_covariates_names
            )
        else:
            past_covariates_cat = []
            past_covariates_real = []

        if data.static_features is not None:
            logger.info(f"\tstatic features:  {data.static_features.columns.to_list()}")
            static_features_df = self.static_feature_pipeline.fit_transform(data.static_features)
            static_features_cat, static_features_real = self._detect_inferred_types(
                static_features_df, original_column_names=data.static_features.columns
            )
        else:
            static_features_cat = []
            static_features_real = []

        if len(data.columns) > 1 or data.static_features is not None:
            logger.info(
                "To learn how to fix incorrectly inferred types, please see documentation for TimeSeriesPredictor.fit "
            )

        self.covariate_metadata = CovariateMetadata(
            static_features_cat=static_features_cat,
            static_features_real=static_features_real,
            known_covariates_real=known_covariates_real,
            past_covariates_real=past_covariates_real,
            known_covariates_cat=known_covariates_cat,
            past_covariates_cat=past_covariates_cat,
        )
        self._is_fit = True

    @staticmethod
    def _check_and_prepare_covariates(
        data: TimeSeriesDataFrame,
        required_column_names: List[str],
        data_frame_name: str,
    ) -> TimeSeriesDataFrame:
        """Select the required columns from the data frame."""
        missing_columns = pd.Index(required_column_names).difference(data.columns)
        if len(missing_columns) > 0:
            raise ValueError(
                f"{len(missing_columns)} columns are missing from {data_frame_name}: {reprlib.repr(missing_columns.to_list())}"
            )
        return data[required_column_names]

    def transform(self, data: TimeSeriesDataFrame, data_frame_name: str = "data") -> TimeSeriesDataFrame:
        """Transform static features and past/known covariates.

        Transformed data is guaranteed to match the specification (same columns / dtypes) of the data seen during fit.
        Extra columns not seen during fitting will be removed.

        If some columns are missing or are incompatible, an exception will be raised.
        """
        assert self._is_fit, f"{self.__class__.__name__} has not been fit yet"
        # Avoid modifying inplace
        data = data.copy(deep=False)
        data = self._check_and_prepare_covariates(
            data, required_column_names=self.required_column_names, data_frame_name=data_frame_name
        )

        if len(self.known_covariates_names) > 0:
            data[self.known_covariates_names] = self.known_covariates_pipeline.transform(
                data[self.known_covariates_names]
            )
        if len(self.past_covariates_names) > 0:
            data[self.past_covariates_names] = self.past_covariates_pipeline.transform(
                data[self.past_covariates_names]
            )

        if self.static_feature_pipeline.is_fit():
            if data.static_features is None:
                raise ValueError(f"Provided {data_frame_name} must contain static_features")
            static_features = self.static_feature_pipeline.transform(data.static_features)
            data.static_features = self._convert_numerical_features_to_float(static_features)
        else:
            data.static_features = None

        return data

    def transform_future_known_covariates(
        self, known_covariates: Optional[TimeSeriesDataFrame]
    ) -> Optional[TimeSeriesDataFrame]:
        assert self._is_fit, f"{self.__class__.__name__} has not been fit yet"
        if len(self.known_covariates_names) > 0:
            assert known_covariates is not None, "known_covariates must be provided at prediction time"
            known_covariates = self._check_and_prepare_covariates(
                known_covariates,
                required_column_names=self.known_covariates_names,
                data_frame_name="known_covariates",
            )
            return self.known_covariates_pipeline.transform(known_covariates)
        else:
            return None

    def fit_transform(self, data: TimeSeriesDataFrame, data_frame_name: str = "data") -> TimeSeriesDataFrame:
        self.fit(data)
        return self.transform(data, data_frame_name=data_frame_name)
