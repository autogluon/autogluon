import logging
from typing import List, Optional

import pandas as pd

from autogluon.features import AbstractFeatureGenerator, AutoMLPipelineFeatureGenerator

from ..state import AnalysisState, StateCheckMixin
from .base import AbstractAnalysis

logger = logging.getLogger(__name__)

__all__ = ["ApplyFeatureGenerator"]


class ApplyFeatureGenerator(AbstractAnalysis, StateCheckMixin):
    """
    This wrapper provides transformed features to all `children` shadowing outer datasets with the updated one after application of FeatureGenerator.

    Parameters
    ----------
    category_to_numbers: bool, default = False
        if `True', then transform `category` variables into their codes. This is useful when wrapped analyses expect numeric values
    feature_generator: Optional[AbstractFeatureGenerator], default = None
        feature generator to use for the transformation. If `None` is provided then `AutoMLPipelineFeatureGenerator` is applied.
    parent: Optional[AbstractAnalysis], default = None
        parent Analysis
    children: Optional[List[AbstractAnalysis]], default None
        wrapped analyses; these will receive sampled `args` during `fit` call
    kwargs

    See also :func:`autogluon.features.AbstractFeatureGenerator`

    Examples
    --------
    >>> from autogluon.eda.analysis.base import BaseAnalysis, Namespace
    >>> import pandas as pd
    >>> import numpy as np
    >>> df_train = pd.DataFrame(...)
    >>> df_test = pd.DataFrame(...)
    >>>
    >>> analysis = BaseAnalysis(train_data=df_train, test_data=df_test, label='D', children=[
    >>>     Namespace(namespace='feature_generator_numbers', children=[
    >>>         ApplyFeatureGenerator(category_to_numbers=True, children=[
    >>>             # SomeAnalysis()  # This analysis will be called with transformed `train_data` and `test_data`
    >>>         ])
    >>>     ]),
    >>>     # SomeAnalysis()  # This analysis will be called with the original features
    >>> ])

    """

    def __init__(
        self,
        parent: Optional[AbstractAnalysis] = None,
        children: Optional[List[AbstractAnalysis]] = None,
        state: Optional[AnalysisState] = None,
        category_to_numbers: bool = False,
        feature_generator: Optional[AbstractFeatureGenerator] = None,
        **kwargs,
    ) -> None:
        super().__init__(parent, children, state, **kwargs)
        self.category_to_numbers = category_to_numbers
        if feature_generator is None:
            feature_generator = AutoMLPipelineFeatureGenerator(
                enable_numeric_features=True,
                enable_categorical_features=True,
                enable_datetime_features=False,
                enable_text_special_features=False,
                enable_text_ngram_features=False,
                enable_raw_text_features=False,
                enable_vision_features=False,
                verbosity=0,
                **kwargs,
            )
        self.feature_generator = feature_generator

    def can_handle(self, state: AnalysisState, args: AnalysisState) -> bool:
        return self.all_keys_must_be_present(args, "train_data", "label")

    def _fit(self, state: AnalysisState, args: AnalysisState, **fit_kwargs) -> None:
        x = args.train_data
        if args.label is not None:
            x = x.drop(columns=args.label)
        self.feature_generator.fit(x)
        self.args["feature_generator"] = True
        for ds, df in self.available_datasets(args):
            x = df
            y = None
            if args.label in df.columns:
                x = df.drop(columns=args.label)
                y = df[args.label]
            x_tx = self.feature_generator.transform(x)
            if self.category_to_numbers:
                for col, dtype in x_tx.dtypes.items():
                    if dtype == "category":
                        x_tx[col] = x_tx[col].cat.codes
            if y is not None:
                x_tx = pd.concat([x_tx, y], axis=1)
            self.args[ds] = x_tx
