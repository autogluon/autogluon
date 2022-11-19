from typing import List, Optional

import pandas as pd

from autogluon.eda import AnalysisState
from autogluon.eda.analysis.base import AbstractAnalysis
from autogluon.eda.state import StateCheckMixin
from autogluon.features import AbstractFeatureGenerator, AutoMLPipelineFeatureGenerator


class ApplyFeatureGenerator(AbstractAnalysis, StateCheckMixin):

    def __init__(self,
                 parent: Optional[AbstractAnalysis] = None,
                 children: List[AbstractAnalysis] = [],
                 state: Optional[AnalysisState] = None,
                 category_to_numbers: bool = False,
                 feature_generator: Optional[AbstractFeatureGenerator] = None,
                 **kwargs) -> None:
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
            )
        self.feature_generator = feature_generator

    def can_handle(self, state: AnalysisState, args: AnalysisState) -> bool:
        return self.all_keys_must_be_present(args, 'train_data', 'label')

    def _fit(self, state: AnalysisState, args: AnalysisState, **fit_kwargs) -> None:
        x = args.train_data.drop(columns=args.label)
        self.feature_generator.fit(x)
        self.args['feature_generator'] = True
        for (ds, df) in self.available_datasets(args):
            x = df
            y = None
            if args.label in df.columns:
                x = df.drop(columns=args.label)
                y = df[args.label]
            x_tx = self.feature_generator.transform(x)
            if self.category_to_numbers:
                for col, dtype in x_tx.dtypes.items():
                    if dtype == 'category':
                        x_tx[col] = x_tx[col].cat.codes
            if y is not None:
                x_tx = pd.concat([x_tx, y], axis=1)
            self.args[ds] = x_tx
