import logging
import reprlib
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import pandas as pd

from autogluon.common.features.types import R_FLOAT, R_INT
from autogluon.features.generators import (
    AsTypeFeatureGenerator,
    CategoryFeatureGenerator,
    IdentityFeatureGenerator,
    PipelineFeatureGenerator,
)
from autogluon.timeseries import TimeSeriesDataFrame

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

    def __init__(self, verbosity: int = 0, minimum_cat_count=2, float_dtype: str = "float32", **kwargs):
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
            verbosity=verbosity,
            **kwargs,
        )
        self.float_dtype = float_dtype

    def _convert_numerical_columns_to_float(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert the dtype of all numerical (float or int) columns to the given float dtype."""
        numeric_columns = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
        return df.astype({col: self.float_dtype for col in numeric_columns})

    def transform(self, X: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        if isinstance(X, TimeSeriesDataFrame):
            X = pd.DataFrame(X)
        return self._convert_numerical_columns_to_float(super().transform(X, *args, **kwargs))

    def fit_transform(self, X: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        # PipelineFeatureGenerator does not use transform() inside fit_transform(), so we need to override both methods
        if isinstance(X, TimeSeriesDataFrame):
            X = pd.DataFrame(X)
        return self._convert_numerical_columns_to_float(super().fit_transform(X, *args, **kwargs))


class TimeSeriesFeatureGenerator:
    """Takes care of preprocessing for static_features and past/known covariates.

    All covariates & static features are converted into either float32 or categorical dtype.
    """

    def __init__(self, target: str, known_covariates_names: List[str], float_dtype: str = "float32"):
        self.target = target
        self.float_dtype = float_dtype
        self._is_fit = False
        self.known_covariates_names = list(known_covariates_names)
        self.past_covariates_names = []
        self.known_covariates_pipeline = ContinuousAndCategoricalFeatureGenerator()
        self.past_covariates_pipeline = ContinuousAndCategoricalFeatureGenerator()
        # Cat features with cat_count=1 are fine in static_features since they are repeated for all time steps in a TS
        self.static_feature_pipeline = ContinuousAndCategoricalFeatureGenerator(minimum_cat_count=1)
        self.covariate_metadata: CovariateMetadata = None

    @property
    def required_column_names(self) -> List[str]:
        return [self.target] + list(self.known_covariates_names) + list(self.past_covariates_names)

    def fit(self, data: TimeSeriesDataFrame) -> None:
        assert not self._is_fit, f"{self.__class__.__name__} has already been fit"

        self.past_covariates_names = []
        for column in data.columns:
            if column != self.target and column not in self.known_covariates_names:
                self.past_covariates_names.append(column)

        self._check_required_columns_are_present(
            data, required_column_names=self.required_column_names, data_frame_name="train_data"
        )

        logger.info("\nProvided data contains following columns:")
        logger.info(f"\ttarget: '{self.target}'")

        if len(self.known_covariates_names) > 0:
            known_covariates_df = self.known_covariates_pipeline.fit_transform(data[self.known_covariates_names])
            logger.info("\tknown_covariates:")
            known_covariates_cat, known_covariates_real = self._detect_and_log_column_types(known_covariates_df)
            self.known_covariates_names = self.known_covariates_pipeline.features_in
        else:
            known_covariates_cat = []
            known_covariates_real = []

        if len(self.past_covariates_names) > 0:
            past_covariates_df = self.past_covariates_pipeline.fit_transform(data[self.past_covariates_names])
            logger.info("\tpast_covariates:")
            past_covariates_cat, past_covariates_real = self._detect_and_log_column_types(past_covariates_df)
            self.past_covariates_names = self.past_covariates_pipeline.features_in
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
        else:
            static_features_cat = []
            static_features_real = []
            ignored_static_features = []

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
        self._is_fit = True

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
        else:
            static_features = None

        return TimeSeriesDataFrame(pd.concat(dfs, axis=1), static_features=static_features)

    def transform_future_known_covariates(
        self, known_covariates: Optional[TimeSeriesDataFrame]
    ) -> Optional[TimeSeriesDataFrame]:
        assert self._is_fit, f"{self.__class__.__name__} has not been fit yet"
        if len(self.known_covariates_names) > 0:
            assert known_covariates is not None, "known_covariates must be provided at prediction time"
            self._check_required_columns_are_present(
                known_covariates, required_column_names=self.known_covariates_names, data_frame_name="known_covariates"
            )
            return TimeSeriesDataFrame(self.known_covariates_pipeline.transform(known_covariates))
        else:
            return None

    def fit_transform(self, data: TimeSeriesDataFrame, data_frame_name: str = "data") -> TimeSeriesDataFrame:
        self.fit(data)
        return self.transform(data, data_frame_name=data_frame_name)

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
