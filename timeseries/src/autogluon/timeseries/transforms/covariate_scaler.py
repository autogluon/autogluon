import logging
from typing import Dict, List, Literal, Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import QuantileTransformer, StandardScaler

from autogluon.timeseries.dataset.ts_dataframe import TimeSeriesDataFrame
from autogluon.timeseries.utils.features import CovariateMetadata
from autogluon.timeseries.utils.warning_filters import warning_filter

logger = logging.getLogger(__name__)


class CovariateScaler:
    """Apply scaling to covariates and static features.

    This can be helpful for deep learning models that assume that the inputs are normalized.
    """

    def __init__(
        self,
        metadata: CovariateMetadata,
        use_known_covariates: bool = True,
        use_past_covariates: bool = True,
        use_static_features: bool = True,
        **kwargs,
    ):
        self.metadata = metadata
        self.use_known_covariates = use_known_covariates
        self.use_past_covariates = use_past_covariates
        self.use_static_features = use_static_features

    def fit_transform(self, data: TimeSeriesDataFrame) -> TimeSeriesDataFrame:
        raise NotImplementedError

    def transform(self, data: TimeSeriesDataFrame) -> TimeSeriesDataFrame:
        raise NotImplementedError

    def transform_known_covariates(
        self, known_covariates: Optional[TimeSeriesDataFrame] = None
    ) -> Optional[TimeSeriesDataFrame]:
        raise NotImplementedError


class GlobalCovariateScaler(CovariateScaler):
    """Applies preprocessing logic similar to tabular's NN_TORCH model to the covariates.

    Performs following preprocessing for real-valued columns:
    - sklearn.preprocessing.QuantileTransform for skewed features
    - passthrough (ignore) boolean features
    - sklearn.preprocessing.StandardScaler for the rest of the features

    Preprocessing is done globally across all items.
    """

    def __init__(
        self,
        metadata: CovariateMetadata,
        use_known_covariates: bool = True,
        use_past_covariates: bool = True,
        use_static_features: bool = True,
        skew_threshold: float = 0.99,
    ):
        super().__init__(metadata, use_known_covariates, use_past_covariates, use_static_features)
        self.skew_threshold = skew_threshold
        self._column_transformers: Optional[Dict[Literal["known", "past", "static"], ColumnTransformer]] = None

    def is_fit(self) -> bool:
        return self._column_transformers is not None

    def fit(self, data: TimeSeriesDataFrame) -> "GlobalCovariateScaler":
        self._column_transformers = {}

        if self.use_known_covariates and len(self.metadata.known_covariates_real) > 0:
            self._column_transformers["known"] = self._get_transformer_for_columns(
                data, columns=self.metadata.known_covariates_real
            )
        if self.use_past_covariates and len(self.metadata.past_covariates_real) > 0:
            self._column_transformers["past"] = self._get_transformer_for_columns(
                data, columns=self.metadata.past_covariates_real
            )
        if self.use_static_features and len(self.metadata.static_features_real) > 0:
            self._column_transformers["static"] = self._get_transformer_for_columns(
                data.static_features, columns=self.metadata.static_features_real
            )

    def fit_transform(self, data: TimeSeriesDataFrame) -> TimeSeriesDataFrame:
        if not self.is_fit():
            self.fit(data=data)
        return self.transform(data=data)

    def transform(self, data: TimeSeriesDataFrame) -> TimeSeriesDataFrame:
        # Copy data to avoid inplace modification
        data = data.copy()
        if "known" in self._column_transformers:
            columns = self.metadata.known_covariates_real
            data[columns] = self._column_transformers["known"].transform(data[columns])

        if "past" in self._column_transformers:
            columns = self.metadata.past_covariates_real
            data[columns] = self._column_transformers["past"].transform(data[columns])

        if "static" in self._column_transformers:
            columns = self.metadata.static_features_real
            data.static_features[columns] = self._column_transformers["static"].transform(
                data.static_features[columns]
            )
        return data

    def transform_known_covariates(
        self, known_covariates: Optional[TimeSeriesDataFrame] = None
    ) -> Optional[TimeSeriesDataFrame]:
        if "known" in self._column_transformers:
            columns = self.metadata.known_covariates_real
            known_covariates = known_covariates.copy()
            known_covariates[columns] = self._column_transformers["known"].transform(known_covariates[columns])
        return known_covariates

    def _get_transformer_for_columns(self, df: pd.DataFrame, columns: List[str]) -> Dict[str, str]:
        """Passthrough bool features, use QuantileTransform for skewed features, and use StandardScaler for the rest.

        The preprocessing logic is similar to the TORCH_NN model from Tabular.
        """
        bool_features = []
        skewed_features = []
        continuous_features = []
        for col in columns:
            if set(df[col].unique()) == set([0, 1]):
                bool_features.append(col)
            elif np.abs(df[col].skew()) > self.skew_threshold:
                skewed_features.append(col)
            else:
                continuous_features.append(col)
        transformers = []
        logger.debug(
            f"\tbool_features: {bool_features}, continuous_features: {continuous_features}, skewed_features: {skewed_features}"
        )
        if continuous_features:
            transformers.append(("scaler", StandardScaler(), continuous_features))
        if skewed_features:
            transformers.append(("skew", QuantileTransformer(output_distribution="normal"), skewed_features))
        with warning_filter():
            column_transformer = ColumnTransformer(transformers=transformers, remainder="passthrough").fit(df[columns])
        return column_transformer


AVAILABLE_COVARIATE_SCALERS = {
    "global": GlobalCovariateScaler,
}


def get_covariate_scaler_from_name(name: Literal["global"], **scaler_kwargs) -> CovariateScaler:
    if name not in AVAILABLE_COVARIATE_SCALERS:
        raise KeyError(
            f"Covariate scaler type {name} not supported. Available scalers: {list(AVAILABLE_COVARIATE_SCALERS)}"
        )
    return AVAILABLE_COVARIATE_SCALERS[name](**scaler_kwargs)
