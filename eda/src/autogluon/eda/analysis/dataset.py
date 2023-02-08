from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Union

import numpy as np
import pandas as pd

from autogluon.common.features.infer_types import get_type_group_map_special, get_type_map_raw
from autogluon.common.features.types import R_BOOL, R_CATEGORY, R_FLOAT, R_INT, R_OBJECT

from ..state import AnalysisState
from .base import AbstractAnalysis

__all__ = [
    "DatasetSummary",
    "RawTypesAnalysis",
    "Sampler",
    "SpecialTypesAnalysis",
    "VariableTypeAnalysis",
    "TrainValidationSplit",
    "ProblemTypeControl",
    "LabelInsightsAnalysis",
]

from autogluon.core.constants import (
    BINARY,
    MULTICLASS,
    PROBLEM_TYPES_CLASSIFICATION,
    PROBLEM_TYPES_REGRESSION,
    REGRESSION,
)
from autogluon.core.utils import generate_train_test_split_combined, infer_problem_type


class Sampler(AbstractAnalysis):
    """
    Sampler is a wrapper that provides sampling capabilites for the wrapped analyses.
    The sampling is performed for all datasets in `args` and passed to all `children` during `fit` call shadowing outer parameters.

    Parameters
    ----------
    sample: Union[None, int, float], default = None
        sample size; if `int`, then row number is used;
        `float` must be between 0.0 and 1.0 and represents fraction of dataset to sample;
        `None` means no sampling
    parent: Optional[AbstractAnalysis], default = None
        parent Analysis
    children: Optional[List[AbstractAnalysis]], default None
        wrapped analyses; these will receive sampled `args` during `fit` call

    Examples
    --------
    >>> from autogluon.eda.analysis.base import BaseAnalysis
    >>> from autogluon.eda.analysis import Sampler
    >>> import pandas as pd
    >>> import numpy as np
    >>>
    >>> df_train = pd.DataFrame(np.random.randint(0, 100, size=(10, 4)), columns=list('ABCD'))
    >>> df_test = pd.DataFrame(np.random.randint(0, 100, size=(20, 4)), columns=list('EFGH'))
    >>> analysis = BaseAnalysis(train_data=df_train, test_data=df_test, children=[
    >>>     Sampler(sample=5, children=[
    >>>         # Analysis here will be performed on a sample of 5 for both train_data and test_data
    >>>     ])
    >>> ])
    """

    def __init__(
        self,
        sample: Union[None, int, float] = None,
        parent: Optional[AbstractAnalysis] = None,
        children: Optional[List[AbstractAnalysis]] = None,
        **kwargs,
    ) -> None:
        super().__init__(parent, children, **kwargs)
        if sample is not None and isinstance(sample, float):
            assert 0.0 < sample < 1.0, "sample must be within the range (0.0, 1.0)"
        self.sample = sample

    def can_handle(self, state: AnalysisState, args: AnalysisState) -> bool:
        return True

    def _fit(self, state: AnalysisState, args: AnalysisState, **fit_kwargs) -> None:
        if self.sample is not None:
            state.sample_size = self.sample
            for ds, df in self.available_datasets(args):
                arg = "n"
                if self.sample is not None and isinstance(self.sample, float):
                    arg = "frac"
                self.args[ds] = df.sample(**{arg: self.sample}, random_state=0)


class ProblemTypeControl(AbstractAnalysis):
    """
    Helper component to control problem type. Autodetect if `problem_type = 'auto'`.

    Parameters
    ----------
    problem_type: str, default = 'auto'
        problem type to use. Valid problem_type values include ['auto', 'binary', 'multiclass', 'regression', 'quantile', 'softclass']
        auto means it will be Auto-detected using AutoGluon methods.
    parent: Optional[AbstractAnalysis], default = None
        parent Analysis
    children: Optional[List[AbstractAnalysis]], default None
        wrapped analyses; these will receive sampled `args` during `fit` call
    kwargs
    """

    def __init__(
        self,
        problem_type: str = "auto",
        parent: Optional[AbstractAnalysis] = None,
        children: Optional[List[AbstractAnalysis]] = None,
        **kwargs,
    ) -> None:
        super().__init__(parent, children, **kwargs)
        valid_problem_types = ["auto"] + PROBLEM_TYPES_REGRESSION + PROBLEM_TYPES_CLASSIFICATION
        assert problem_type in valid_problem_types, f"Valid problem_type values include {valid_problem_types}"
        self.problem_type = problem_type

    def can_handle(self, state: AnalysisState, args: AnalysisState) -> bool:
        return self.all_keys_must_be_present(args, "train_data", "label")

    def _fit(self, state: AnalysisState, args: AnalysisState, **fit_kwargs) -> None:
        if self.problem_type == "auto":
            state.problem_type = infer_problem_type(args.train_data[args.label], silent=True)
        else:
            state.problem_type = self.problem_type


class TrainValidationSplit(AbstractAnalysis):
    """
    This wrapper splits `train_data` into training and validation sets stored in `train_data` and `val_data` for the wrapped analyses.
    The split is performed for datasets in `args` and passed to all `children` during `fit` call shadowing outer parameters.

    This component requires :py:class:`~autogluon.eda.visualization.dataset.ProblemTypeControl` present in the analysis call to set `problem_type`.

    Parameters
    ----------
    val_size: float, default = 0.3
        fraction of training set to be assigned as validation set during the split.
    parent: Optional[AbstractAnalysis], default = None
        parent Analysis
    children: Optional[List[AbstractAnalysis]], default None
        wrapped analyses; these will receive sampled `args` during `fit` call
    kwargs

    Examples
    --------
    >>> from autogluon.eda.analysis.base import BaseAnalysis
    >>> from autogluon.eda.analysis import Sampler
    >>> import pandas as pd
    >>> import numpy as np
    >>>
    >>> df_train = pd.DataFrame(np.random.randint(0, 100, size=(100, 4)), columns=list("ABCD"))
    >>> analysis = BaseAnalysis(train_data=df_train, label="D", children=[
    >>>         Namespace(namespace="ns_val_split_specified", children=[
    >>>             ProblemTypeControl(),
    >>>             TrainValidationSplit(val_pct=0.4, children=[
    >>>                 # This analysis sees 60/40 split of df_train between train_data and val_data
    >>>                 SomeAnalysis()
    >>>             ])
    >>>         ]),
    >>>         Namespace(namespace="ns_val_split_default", children=[
    >>>             ProblemTypeControl(),
    >>>             TrainValidationSplit(children=[
    >>>                 # This analysis sees 70/30 split (default) of df_train between train_data and val_data
    >>>                 SomeAnalysis()
    >>>             ])
    >>>         ]),
    >>>         Namespace(namespace="ns_no_split", children=[
    >>>                 # This analysis sees only original train_data
    >>>             SomeAnalysis()
    >>>         ]),
    >>>     ],
    >>> )
    >>>
    >>> state = analysis.fit()
    >>>

    See Also
    --------
    :py:class:`~autogluon.eda.visualization.dataset.ProblemTypeControl`

    """

    def __init__(
        self,
        val_size: float = 0.3,
        parent: Optional[AbstractAnalysis] = None,
        children: Optional[List[AbstractAnalysis]] = None,
        **kwargs,
    ) -> None:
        super().__init__(parent, children, **kwargs)

        assert 0 < val_size < 1.0, "val_size must be between 0 and 1"
        self.val_size = val_size

    def can_handle(self, state: AnalysisState, args: AnalysisState) -> bool:
        return self.all_keys_must_be_present(state, "problem_type")

    def _fit(self, state: AnalysisState, args: AnalysisState, **fit_kwargs) -> None:
        train_data, val_data = generate_train_test_split_combined(
            args.train_data, args.label, state.problem_type, test_size=self.val_size, **self.args
        )
        self.args["train_data"] = train_data
        self.args["val_data"] = val_data


class DatasetSummary(AbstractAnalysis):
    """
    Generates dataset summary including counts, number of unique elements, most frequent, dtypes and 7-figure summary (std/mean/min/max/quartiles)

    Examples
    --------
    >>> import autogluon.eda.analysis as eda
    >>> import autogluon.eda.visualization as viz
    >>> import autogluon.eda.auto as auto
    >>> state = auto.analyze(
    >>>     train_data=..., label=..., return_state=True,
    >>>     anlz_facets=[
    >>>         eda.dataset.DatasetSummary(),
    >>>     ],
    >>>     viz_facets=[
    >>>         viz.dataset.DatasetStatistics()
    >>>     ]
    >>> )

    See Also
    --------
    :py:class:`~autogluon.eda.visualization.dataset.DatasetStatistics`
    """

    def can_handle(self, state: AnalysisState, args: AnalysisState) -> bool:
        return True

    def _fit(self, state: AnalysisState, args: AnalysisState, **fit_kwargs) -> None:
        s = {}
        for ds, df in self.available_datasets(args):
            summary = df.describe(include="all").T
            summary = summary.join(pd.DataFrame({"dtypes": df.dtypes}))
            summary["unique"] = args[ds].nunique()
            summary["count"] = summary["count"].astype(int)
            summary = summary.sort_index()
            s[ds] = summary.to_dict()
        state.dataset_stats = s


class RawTypesAnalysis(AbstractAnalysis):
    """
    Infers autogluon raw types for the column.

    Examples
    --------
    >>> import autogluon.eda.analysis as eda
    >>> import autogluon.eda.visualization as viz
    >>> import autogluon.eda.auto as auto
    >>> state = auto.analyze(
    >>>     train_data=..., label=..., return_state=True,
    >>>     anlz_facets=[
    >>>         eda.dataset.RawTypesAnalysis(),
    >>>     ],
    >>>     viz_facets=[
    >>>         viz.dataset.DatasetStatistics()
    >>>     ]
    >>> )

    See Also
    --------
    :py:class:`~autogluon.eda.visualization.dataset.DatasetStatistics`
    """

    def can_handle(self, state: AnalysisState, args: AnalysisState) -> bool:
        return True

    def _fit(self, state: AnalysisState, args: AnalysisState, **fit_kwargs) -> None:
        state.raw_type = {}
        for ds, df in self.available_datasets(args):
            state.raw_type[ds] = get_type_map_raw(df)


class VariableTypeAnalysis(AbstractAnalysis):
    """
    Infers variable types for the column: numeric vs category.

    This analysis depends on :func:`RawTypesAnalysis`.

    Parameters
    ----------
    numeric_as_categorical_threshold: int, default = 20
        if numeric column has less than this value, then the variable should be considered as categorical
    parent: Optional[AbstractAnalysis], default = None
        parent Analysis
    children: Optional[List[AbstractAnalysis]], default None
        wrapped analyses; these will receive sampled `args` during `fit` call

    Examples
    --------
    >>> import autogluon.eda.analysis as eda
    >>> import autogluon.eda.visualization as viz
    >>> import autogluon.eda.auto as auto
    >>> state = auto.analyze(
    >>>     train_data=..., label=..., return_state=True,
    >>>     anlz_facets=[
    >>>         eda.dataset.RawTypesAnalysis(),
    >>>         eda.dataset.VariableTypeAnalysis(),
    >>>     ],
    >>>     viz_facets=[
    >>>         viz.dataset.DatasetStatistics()
    >>>     ]
    >>> )

    See Also
    --------
    :py:class:`~autogluon.eda.analysis.dataset.RawTypesAnalysis`
    :py:class:`~autogluon.eda.visualization.dataset.DatasetStatistics`
    """

    def __init__(
        self,
        parent: Union[None, AbstractAnalysis] = None,
        children: Optional[List[AbstractAnalysis]] = None,
        numeric_as_categorical_threshold: int = 20,
        **kwargs,
    ) -> None:
        super().__init__(parent, children, **kwargs)
        self.numeric_as_categorical_threshold = numeric_as_categorical_threshold

    def can_handle(self, state: AnalysisState, args: AnalysisState) -> bool:
        return self.all_keys_must_be_present(state, "raw_type")

    def _fit(self, state: AnalysisState, args: AnalysisState, **fit_kwargs) -> None:
        state.variable_type = {}
        for ds, df in self.available_datasets(args):
            state.variable_type[ds] = {
                c: self.map_raw_type_to_feature_type(c, t, df, self.numeric_as_categorical_threshold)
                for c, t in state.raw_type[ds].items()
            }

    @staticmethod
    def map_raw_type_to_feature_type(
        col: Optional[str], raw_type: str, df: pd.DataFrame, numeric_as_categorical_threshold: int = 20
    ) -> Union[None, str]:
        if col is None:
            return None
        elif df[col].nunique() <= numeric_as_categorical_threshold:
            return "category"
        elif raw_type in [R_INT, R_FLOAT]:
            return "numeric"
        elif raw_type in [R_OBJECT, R_CATEGORY, R_BOOL]:
            return "category"
        else:
            return None


class SpecialTypesAnalysis(AbstractAnalysis):
    """
    Infers autogluon special types for the column (i.e. text).

    Examples
    --------
    >>> import autogluon.eda.analysis as eda
    >>> import autogluon.eda.visualization as viz
    >>> import autogluon.eda.auto as auto
    >>> state = auto.analyze(
    >>>     train_data=..., label=..., return_state=True,
    >>>     anlz_facets=[
    >>>         eda.dataset.SpecialTypesAnalysis(),
    >>>     ],
    >>>     viz_facets=[
    >>>         viz.dataset.DatasetStatistics()
    >>>     ]
    >>> )

    See Also
    --------
    :py:class:`~autogluon.eda.visualization.dataset.DatasetStatistics`
    """

    def can_handle(self, state: AnalysisState, args: AnalysisState) -> bool:
        return True

    def _fit(self, state: AnalysisState, args: AnalysisState, **fit_kwargs) -> None:
        state.special_types = {}
        for ds, df in self.available_datasets(args):
            state.special_types[ds] = self.infer_special_types(df)

    @staticmethod
    def infer_special_types(ds):
        special_types: Dict[str, Set[str]] = {}
        for t, cols in get_type_group_map_special(ds).items():
            for col in cols:
                if col not in special_types:
                    special_types[col] = set()
                special_types[col].add(t)
        result: Dict[str, str] = {}
        for col, types in special_types.items():
            result[col] = ", ".join(sorted(types))
        return result


class LabelInsightsAnalysis(AbstractAnalysis):
    """
     Analyze label for insights:

     - classification: low cardinality classes detection
     - classification: classes present in test data, but not in the train data
     - regression: out-of-domain labels detection

     Note: this Analysis requires `problem_type` present in state.
     It can be detected/set via :py:class:`~autogluon.eda.analysis.dataset.ProblemTypeControl` component

    Parameters
     ----------
     low_cardinality_classes_threshold: int, default = 50
         Minimum class instances present in the dataset to consider marking a class as low-cardinality
     regression_ood_threshold: float, default = 0.01
         mark results as out-of-domain when test label range in regression task is beyond train data range + regression_ood_threshold margin,
         This is performed because some algorithms can't extrapolate beyond training data.
     class_imbalance_ratio_threshold: float, default = 0.4
        minority class proportion to detect as imbalance.
     parent: Optional[AbstractAnalysis], default = None
         parent Analysis
     children: Optional[List[AbstractAnalysis]], default None
         wrapped analyses; these will receive sampled `args` during `fit` call
     state: AnalysisState
         state object to perform check on

     Examples
     --------
     >>> import autogluon.eda.analysis as eda
     >>> import autogluon.eda.visualization as viz
     >>> import autogluon.eda.auto as auto
     >>> auto.analyze(
     >>> auto.analyze(train_data=..., test_data=..., label=..., anlz_facets=[
     >>>     eda.dataset.ProblemTypeControl(),
     >>>     eda.dataset.LabelInsightsAnalysis(low_cardinality_classes_threshold=50, regression_ood_threshold=0.01),
     >>> ], viz_facets=[
     >>>     viz.dataset.LabelInsightsVisualization()
     >>> ])

     See Also
     --------
     :py:class:`~autogluon.eda.analysis.dataset.ProblemTypeControl`
     :py:class:`~autogluon.eda.visualization.dataset.LabelInsightsVisualization`

    """

    def __init__(
        self,
        low_cardinality_classes_threshold: int = 50,
        regression_ood_threshold: float = 0.01,
        class_imbalance_ratio_threshold: float = 0.4,
        parent: Optional[AbstractAnalysis] = None,
        children: Optional[List[AbstractAnalysis]] = None,
        state: Optional[AnalysisState] = None,
        **kwargs,
    ) -> None:
        super().__init__(parent, children, state, **kwargs)
        assert low_cardinality_classes_threshold > 0, "low_cardinality_classes_threshold must be greater than 0"
        self.low_cardinality_classes_threshold = low_cardinality_classes_threshold

        assert 0 < class_imbalance_ratio_threshold < 1, "class_imbalance_ratio_threshold must be between 0 and 1"
        self.class_imbalance_ratio_threshold = class_imbalance_ratio_threshold

        assert 0 < regression_ood_threshold < 1, "regression_ood_threshold must be between 0 and 1"
        self.regression_ood_threshold = regression_ood_threshold

        assert regression_ood_threshold >= 0, "regression_ood_threshold must be non-negative"
        self.regression_ood_threshold = regression_ood_threshold

    def can_handle(self, state: AnalysisState, args: AnalysisState) -> bool:
        return self.all_keys_must_be_present(args, "train_data", "label") and self.all_keys_must_be_present(
            state, "problem_type"
        )

    def _fit(self, state: AnalysisState, args: AnalysisState, **fit_kwargs) -> None:
        label = args.label
        train_data = args.train_data

        s: Dict[str, Any] = {}

        if state.problem_type in [BINARY, MULTICLASS]:
            label_counts = train_data[label].value_counts()
            minority_class = label_counts[label_counts == label_counts.min()].index.values[0]
            majority_class = label_counts[label_counts == label_counts.max()].index.values[0]
            minority_class_imbalance_ratio = min(label_counts) / max(label_counts)

            # Low-cardinality class detection
            label_counts = label_counts[label_counts < self.low_cardinality_classes_threshold].to_dict()
            if len(label_counts) > 0:
                s["low_cardinality_classes"] = {
                    "instances": label_counts,
                    "threshold": self.low_cardinality_classes_threshold,
                }

            # Class imbalance detection
            if minority_class_imbalance_ratio < self.class_imbalance_ratio_threshold:
                s["minority_class_imbalance"] = {
                    "majority_class": majority_class,
                    "minority_class": minority_class,
                    "ratio": minority_class_imbalance_ratio,
                }

            # Classes not found in test_data
            if self._test_data_with_label_present(args, label):
                train_labels = set(train_data[label].unique())
                test_labels = set(args.test_data[label].unique())
                if sorted(train_labels) != sorted(test_labels):
                    missing_classes = test_labels.difference(train_labels)
                    s["not_present_in_train"] = missing_classes
        elif (state.problem_type in [REGRESSION]) and self._test_data_with_label_present(args, label):
            # Out-of-domain range detection
            test_data = args.test_data
            label_min, label_max = np.min(train_data[label]), np.max(train_data[label])
            padding = np.abs(label_max - label_min) * self.regression_ood_threshold
            df_ood = args.test_data[
                (test_data[label] < label_min - padding) | (test_data[label] > label_max + padding)
            ]

            if len(df_ood) > 0:
                s["ood"] = {
                    "count": len(df_ood),
                    "train_range": [label_min, label_max],
                    "test_range": [np.min(test_data[label]), np.max(test_data[label])],
                    "threshold": self.regression_ood_threshold,
                }
        if len(s) > 0:
            state.label_insights = s

    def _test_data_with_label_present(self, args, label):
        return (args.test_data is not None) and (label in args.test_data.columns)
