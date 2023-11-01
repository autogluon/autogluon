import logging
import reprlib
from dataclasses import dataclass, field
from typing import List, Optional

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
        self.static_feature_pipeline = ContinuousAndCategoricalFeatureGenerator()
        self.covariate_metadata: CovariateMetadata = None

    @property
    def required_column_names(self) -> List[str]:
        return [self.target] + list(self.known_covariates_names) + list(self.past_covariates_names)

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
        logger.info(f"\ttarget:           '{self.target}'")
        if len(self.known_covariates_names) > 0:
            logger.info(f"\tknown covariates: {self.known_covariates_names}")
        if len(self.past_covariates_names) > 0:
            logger.info(f"\tpast covariates:  {self.past_covariates_names}")

        static_features_cat = []
        static_features_real = []
        if data.static_features is not None:
            static = self.static_feature_pipeline.fit_transform(data.static_features)
            static = self._convert_numerical_features_to_float(static)

            unused = []
            for col_name in data.static_features.columns:
                if col_name in static.columns and static[col_name].dtype == "category":
                    static_features_cat.append(col_name)
                elif col_name in static.columns and static[col_name].dtype == np.float64:
                    static_features_real.append(col_name)
                else:
                    unused.append(col_name)

            logger.info("Following types of static features have been inferred:")
            logger.info(f"\tcategorical:        {static_features_cat}")
            logger.info(f"\tcontinuous (float): {static_features_real}")
            if len(unused) > 0:
                logger.info(f"\tremoved (uninformative columns): {unused}")
            logger.info(
                "To learn how to fix incorrectly inferred types, please see documentation for TimeSeriesPredictor.fit "
            )

        self.covariate_metadata = CovariateMetadata(
            static_features_cat=static_features_cat,
            static_features_real=static_features_real,
            known_covariates_real=self.known_covariates_names,
            past_covariates_real=self.past_covariates_names,
            # TODO: Categorical time-varying covariates are not yet supported
            known_covariates_cat=[],
            past_covariates_cat=[],
        )
        self._is_fit = True

    @staticmethod
    def _check_and_prepare_covariates(
        data: TimeSeriesDataFrame,
        required_column_names: List[str],
        data_frame_name: str,
    ) -> TimeSeriesDataFrame:
        """Select the required dataframe columns and convert them to float64 dtype."""
        missing_columns = pd.Index(required_column_names).difference(data.columns)
        if len(missing_columns) > 0:
            raise ValueError(
                f"{len(missing_columns)} columns are missing from {data_frame_name}: {reprlib.repr(missing_columns.to_list())}"
            )
        data = data[required_column_names]
        try:
            data = data.astype(np.float64)
        except ValueError:
            raise ValueError(
                f"Columns in {data_frame_name} must all have numeric (float or int) dtypes, "
                f"but in provided data they have dtypes {data.dtypes}"
            )
        return data

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
            data=data,
            required_column_names=self.required_column_names,
            data_frame_name=data_frame_name,
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
            return self._check_and_prepare_covariates(
                known_covariates,
                required_column_names=self.known_covariates_names,
                data_frame_name="known_covariates",
            )
        else:
            return None

    def fit_transform(self, data: TimeSeriesDataFrame, data_frame_name: str = "data") -> TimeSeriesDataFrame:
        self.fit(data)
        return self.transform(data, data_frame_name=data_frame_name)
