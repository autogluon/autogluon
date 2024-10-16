import logging
import reprlib
import time
from dataclasses import dataclass, field
from typing import Any, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd

from autogluon.common.features.types import R_FLOAT, R_INT
from autogluon.features.generators import (
    AsTypeFeatureGenerator,
    CategoryFeatureGenerator,
    IdentityFeatureGenerator,
    PipelineFeatureGenerator,
)
from autogluon.timeseries.dataset.ts_dataframe import ITEMID, TimeSeriesDataFrame
from autogluon.timeseries.utils.warning_filters import warning_filter

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

    @property
    def static_features(self) -> List[str]:
        return self.static_features_cat + self.static_features_real

    @property
    def known_covariates(self) -> List[str]:
        return self.known_covariates_cat + self.known_covariates_real

    @property
    def past_covariates(self) -> List[str]:
        return self.past_covariates_cat + self.past_covariates_real

    @property
    def covariates(self) -> List[str]:
        return self.known_covariates + self.past_covariates

    @property
    def covariates_real(self) -> List[str]:
        return self.known_covariates_real + self.past_covariates_real

    @property
    def covariates_cat(self) -> List[str]:
        return self.known_covariates_cat + self.past_covariates_cat

    @property
    def real_features(self) -> List[str]:
        return self.static_features_real + self.covariates_real

    @property
    def cat_features(self) -> List[str]:
        return self.static_features_cat + self.covariates_cat

    @property
    def all_features(self) -> List[str]:
        return self.static_features + self.covariates


class ContinuousAndCategoricalFeatureGenerator(PipelineFeatureGenerator):
    """Generates categorical and continuous features for time series models.

    Imputes missing categorical features with the most frequent value in the training set.
    """

    def __init__(self, verbosity: int = 0, minimum_cat_count=2, float_dtype: str = "float64", **kwargs):
        generators = [
            CategoryFeatureGenerator(minimum_cat_count=minimum_cat_count, fillna="mode"),
            IdentityFeatureGenerator(infer_features_in_args={"valid_raw_types": [R_INT, R_FLOAT]}),
        ]
        super().__init__(
            generators=[generators],
            post_generators=[],
            pre_generators=[AsTypeFeatureGenerator(convert_bool=False)],
            pre_enforce_types=False,
            pre_drop_useless=False,
            reset_index=False,
            verbosity=verbosity,
            **kwargs,
        )
        self.float_dtype = float_dtype

    # def _convert_numerical_columns_to_float(self, df: pd.DataFrame) -> pd.DataFrame:
    #     """Convert the dtype of all numerical (float or int) columns to the given float dtype."""
    #     numeric_columns = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    #     return df.astype({col: self.float_dtype for col in numeric_columns})

    def transform(self, X: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        return super().transform(X, *args, **kwargs)

    def fit_transform(self, X: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        # PipelineFeatureGenerator does not use transform() inside fit_transform(), so we need to override both methods
        transformed = super().fit_transform(X, *args, **kwargs)
        # Ignore the '__dummy__' feature generated by PipelineFeatureGenerator if none of the features are informative
        if "__dummy__" in transformed.columns:
            transformed = transformed.drop(columns=["__dummy__"], inplace=True)
        return transformed


class TimeSeriesFeatureGenerator:
    """Takes care of preprocessing for static_features and past/known covariates.

    All covariates & static features are converted into either float64 or categorical dtype.

    Missing values in the target column are left as-is but missing values in static features & covariates are imputed.
    Imputation logic is as follows:
    1. For all categorical columns (static, past, known), we fill missing values with the mode of the training set.
    2. For real static features, we impute missing values with the median of the training set.
    3. For real covariates (past, known), we ffill + bfill within each time series. If for some time series all
        covariate values are missing, we fill them with the median of the training set.
    """

    def __init__(
        self,
        target: str,
        known_covariates_names: List[str],
        float_dtype: str = "float64",
        num_samples: Optional[int] = 20_000,
    ):
        self.target = target
        self.float_dtype = float_dtype
        self.num_samples = num_samples

        self._is_fit = False
        self.known_covariates_names = list(known_covariates_names)
        self.past_covariates_names = []
        self.known_covariates_pipeline = ContinuousAndCategoricalFeatureGenerator(float_dtype=float_dtype)
        self.past_covariates_pipeline = ContinuousAndCategoricalFeatureGenerator(float_dtype=float_dtype)
        # Cat features with cat_count=1 are fine in static_features since they are repeated for all time steps in a TS
        self.static_feature_pipeline = ContinuousAndCategoricalFeatureGenerator(
            minimum_cat_count=1, float_dtype=float_dtype
        )
        self.covariate_metadata: CovariateMetadata = None
        self._train_covariates_real_median: Optional[pd.Series] = None
        self._train_static_real_median: Optional[pd.Series] = None

    @property
    def required_column_names(self) -> List[str]:
        return [self.target] + list(self.known_covariates_names) + list(self.past_covariates_names)

    def fit(self, data: TimeSeriesDataFrame) -> None:
        self.fit_transform(data)

    def fit_transform(self, data: TimeSeriesDataFrame) -> TimeSeriesDataFrame:
        assert not self._is_fit, f"{self.__class__.__name__} has already been fit"

        self.past_covariates_names = []
        for column in data.columns:
            if column != self.target and column not in self.known_covariates_names:
                self.past_covariates_names.append(column)

        self._check_required_columns_are_present(
            data, required_column_names=self.required_column_names, data_frame_name="train_data"
        )

        df = pd.DataFrame(data)
        index = df.index
        df.reset_index(drop=True, inplace=True)

        dfs_to_concat = [df[[self.target]]]

        logger.info("\nProvided data contains following columns:")
        logger.info(f"\ttarget: '{self.target}'")

        if len(self.known_covariates_names) > 0:
            known_covariates_df = self.known_covariates_pipeline.fit_transform(df[self.known_covariates_names])
            logger.info("\tknown_covariates:")
            known_covariates_cat, known_covariates_real = self._detect_and_log_column_types(known_covariates_df)
            self.known_covariates_names = self.known_covariates_pipeline.features_in
            dfs_to_concat.append(known_covariates_df)
        else:
            known_covariates_cat = []
            known_covariates_real = []

        if len(self.past_covariates_names) > 0:
            past_covariates_df = self.past_covariates_pipeline.fit_transform(df[self.past_covariates_names])
            logger.info("\tpast_covariates:")
            past_covariates_cat, past_covariates_real = self._detect_and_log_column_types(past_covariates_df)
            self.past_covariates_names = self.past_covariates_pipeline.features_in
            dfs_to_concat.append(past_covariates_df)
        else:
            past_covariates_cat = []
            past_covariates_real = []

        ignored_covariates = data.columns.difference(
            [self.target] + self.known_covariates_names + self.past_covariates_names
        )

        if data.static_features is not None:
            static_features_df = self.static_feature_pipeline.fit_transform(data.static_features)
            logger.info("\tstatic_features:")
            static_features_cat, static_features_real = self._detect_and_log_column_types(static_features_df)
            ignored_static_features = data.static_features.columns.difference(self.static_feature_pipeline.features_in)
            self._train_static_real_median = data.static_features[static_features_real].median()
        else:
            static_features_cat = []
            static_features_real = []
            ignored_static_features = []
            static_features_df = None

        if len(ignored_covariates) > 0 or len(ignored_static_features) > 0:
            logger.info("\nAutoGluon will ignore following non-numeric/non-informative columns:")
            if len(ignored_covariates) > 0:
                logger.info(f"\tignored covariates:      {list(ignored_covariates)}")
            if len(ignored_static_features) > 0:
                logger.info(f"\tignored static_features: {list(ignored_static_features)}")

        if len(data.columns) > 1 or data.static_features is not None:
            logger.info(
                "\nTo learn how to fix incorrectly inferred types, please see documentation for TimeSeriesPredictor.fit"
            )

        self.covariate_metadata = CovariateMetadata(
            known_covariates_cat=known_covariates_cat,
            known_covariates_real=known_covariates_real,
            past_covariates_cat=past_covariates_cat,
            past_covariates_real=past_covariates_real,
            static_features_cat=static_features_cat,
            static_features_real=static_features_real,
        )
        if len(dfs_to_concat) == 1:
            df_out = dfs_to_concat[0]
        else:
            df_out = pd.concat(dfs_to_concat, axis=1, copy=False)
        df_out.index = index

        if self.num_samples is not None and len(df) > self.num_samples:
            df = df.sample(n=self.num_samples, replace=True)
        self._train_covariates_real_median = df[self.covariate_metadata.covariates_real].median()
        self._is_fit = True

        return TimeSeriesDataFrame(df_out, static_features=static_features_df)

    def transform(self, data: TimeSeriesDataFrame, data_frame_name: str = "data") -> TimeSeriesDataFrame:
        """Transform static features and past/known covariates.

        Transformed data is guaranteed to match the specification (same columns / dtypes) of the data seen during fit.
        Extra columns not seen during fitting will be removed.

        If some columns are missing or are incompatible, an exception will be raised.
        """
        assert self._is_fit, f"{self.__class__.__name__} has not been fit yet"
        self._check_required_columns_are_present(
            data, required_column_names=self.required_column_names, data_frame_name=data_frame_name
        )
        dfs = [data[[self.target]]]

        if len(self.known_covariates_names) > 0:
            dfs.append(self.known_covariates_pipeline.transform(data[self.known_covariates_names]))

        if len(self.past_covariates_names) > 0:
            dfs.append(self.past_covariates_pipeline.transform(data[self.past_covariates_names]))

        if self.static_feature_pipeline.is_fit():
            if data.static_features is None:
                raise ValueError(f"Provided {data_frame_name} must contain static_features")
            static_features = self.static_feature_pipeline.transform(data.static_features)
            static_real_names = self.covariate_metadata.static_features_real
            # Fill missing static_features_real with the median of the training set
            if static_real_names and static_features[static_real_names].isna().any(axis=None):
                static_features[static_real_names] = static_features[static_real_names].fillna(
                    self._train_static_real_median
                )
        else:
            static_features = None

        ts_df = TimeSeriesDataFrame(pd.concat(dfs, axis=1), static_features=static_features)

        covariates_names = self.covariate_metadata.covariates
        if len(covariates_names) > 0:
            # ffill + bfill covariates that have at least some observed values
            ts_df[covariates_names] = ts_df[covariates_names].fill_missing_values()
            # If for some items covariates consist completely of NaNs, fill them with median of training data
            if ts_df[covariates_names].isna().any(axis=None):
                ts_df[covariates_names] = ts_df[covariates_names].fillna(self._train_covariates_real_median)

        return ts_df

    def transform_future_known_covariates(
        self, known_covariates: Optional[TimeSeriesDataFrame]
    ) -> Optional[TimeSeriesDataFrame]:
        assert self._is_fit, f"{self.__class__.__name__} has not been fit yet"
        if len(self.known_covariates_names) > 0:
            assert known_covariates is not None, "known_covariates must be provided at prediction time"
            self._check_required_columns_are_present(
                known_covariates, required_column_names=self.known_covariates_names, data_frame_name="known_covariates"
            )
            known_covariates = TimeSeriesDataFrame(self.known_covariates_pipeline.transform(known_covariates))
            # ffill + bfill covariates that have at least some observed values
            known_covariates = known_covariates.fill_missing_values()
            # If for some items covariates consist completely of NaNs, fill them with median of training data
            if known_covariates.isna().any(axis=None):
                known_covariates = known_covariates.fillna(self._train_covariates_real_median)
            return known_covariates
        else:
            return None

    @staticmethod
    def _detect_and_log_column_types(transformed_df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """Log & return names of categorical and real-valued columns in the DataFrame."""
        cat_column_names = []
        real_column_names = []
        for column_name, column_dtype in transformed_df.dtypes.items():
            if isinstance(column_dtype, pd.CategoricalDtype):
                cat_column_names.append(column_name)
            elif pd.api.types.is_numeric_dtype(column_dtype):
                real_column_names.append(column_name)

        logger.info(f"\t\tcategorical:        {reprlib.repr(cat_column_names)}")
        logger.info(f"\t\tcontinuous (float): {reprlib.repr(real_column_names)}")
        return cat_column_names, real_column_names

    @staticmethod
    def _check_required_columns_are_present(
        data: TimeSeriesDataFrame, required_column_names: List[str], data_frame_name: str
    ) -> None:
        missing_columns = pd.Index(required_column_names).difference(data.columns)
        if len(missing_columns) > 0:
            raise ValueError(
                f"{len(missing_columns)} columns are missing from {data_frame_name}: {reprlib.repr(missing_columns.to_list())}"
            )


class AbstractFeatureImportanceTransform:
    """Abstract class for transforms that replace a given feature with dummy or shuffled values,
    for use in feature importance operations.
    """

    def __init__(
        self,
        covariate_metadata: CovariateMetadata,
        prediction_length: int,
        **kwargs,
    ):
        self.covariate_metadata: CovariateMetadata = covariate_metadata
        self.prediction_length: int = prediction_length

    def _transform_series(self, data: pd.Series, is_categorical: bool, **kwargs) -> TimeSeriesDataFrame:
        """Transforms a series with the same index as the pandas DataFrame"""
        raise NotImplementedError

    def transform(self, data: TimeSeriesDataFrame, feature_name: str, **kwargs) -> TimeSeriesDataFrame:
        if feature_name not in self.covariate_metadata.all_features:
            raise ValueError(f"Target feature {feature_name} not found in covariate metadata")

        # feature transform works on a shallow copy of the main time series data frame
        # but a deep copy of the static features.
        data = data.copy(deep=False)

        is_categorical = feature_name in self.covariate_metadata.cat_features

        if feature_name in self.covariate_metadata.past_covariates:
            # we'll have to work on the history of the data alone
            data[feature_name] = data[feature_name].copy()
            feature_data = data[feature_name].groupby(level=ITEMID, sort=False).head(-self.prediction_length)
            # Silence spurious FutureWarning raised by DataFrame.update https://github.com/pandas-dev/pandas/issues/57124
            with warning_filter():
                data[feature_name].update(self._transform_series(feature_data, is_categorical=is_categorical))
        elif feature_name in self.covariate_metadata.static_features:
            feature_data = data.static_features[feature_name].copy()
            feature_data.reset_index(drop=True, inplace=True)
            data.static_features[feature_name] = self._transform_static_series(
                feature_data, is_categorical=is_categorical
            )
        else:  # known covariates
            data[feature_name] = self._transform_series(data[feature_name], is_categorical=is_categorical)

        return data


class PermutationFeatureImportanceTransform(AbstractFeatureImportanceTransform):
    """Naively shuffles a given feature."""

    def __init__(
        self,
        covariate_metadata: CovariateMetadata,
        prediction_length: int,
        random_seed: Optional[int] = None,
        shuffle_type: Literal["itemwise", "naive"] = "itemwise",
        **kwargs,
    ):
        super().__init__(covariate_metadata, prediction_length, **kwargs)
        self.shuffle_type = shuffle_type
        self.random_seed = random_seed

    def _transform_static_series(self, feature_data: pd.Series, is_categorical: bool) -> Any:
        return feature_data.sample(frac=1, random_state=self.random_seed).values

    def _transform_series(self, feature_data: pd.Series, is_categorical: bool) -> pd.Series:
        # set random state once to shuffle 'independently' for different items
        rng = np.random.RandomState(self.random_seed)

        if self.shuffle_type == "itemwise":
            return feature_data.groupby(level=ITEMID, sort=False).transform(
                lambda x: x.sample(frac=1, random_state=rng).values
            )
        elif self.shuffle_type == "naive":
            return pd.Series(feature_data.sample(frac=1, random_state=rng).values, index=feature_data.index)


class ConstantReplacementFeatureImportanceTransform(AbstractFeatureImportanceTransform):
    """Replaces a target feature with the median if it's a real-valued feature, and the mode if it's a
    categorical feature."""

    def __init__(
        self,
        covariate_metadata: CovariateMetadata,
        prediction_length: int,
        real_value_aggregation: Literal["mean", "median"] = "mean",
        **kwargs,
    ):
        super().__init__(covariate_metadata, prediction_length, **kwargs)
        self.real_value_aggregation = real_value_aggregation

    def _transform_static_series(self, feature_data: pd.Series, is_categorical: bool) -> Any:
        return feature_data.mode()[0] if is_categorical else feature_data.agg(self.real_value_aggregation)

    def _transform_series(self, feature_data: pd.Series, is_categorical: bool) -> pd.Series:
        if is_categorical:
            return feature_data.groupby(level=ITEMID, sort=False).transform(lambda x: x.mode()[0])
        else:
            return feature_data.groupby(level=ITEMID, sort=False).transform(self.real_value_aggregation)
