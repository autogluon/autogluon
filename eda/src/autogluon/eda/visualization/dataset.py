from typing import Any, Dict, List, Optional

import pandas as pd
from pandas import DataFrame

from ..state import AnalysisState
from .base import AbstractVisualization
from .jupyter import JupyterMixin

__all__ = ["DatasetStatistics", "DatasetTypeMismatch", "LabelInsightsVisualization"]


class DatasetStatistics(AbstractVisualization, JupyterMixin):
    """
    Display aggregate dataset statistics and dataset-level information.

    The report is a composite view of combination of performed analyses: :py:class:`~autogluon.eda.analysis.dataset.DatasetSummary`,
    :py:class:`~autogluon.eda.analysis.dataset.RawTypesAnalysis`, :py:class:`~autogluon.eda.analysis.dataset.VariableTypeAnalysis`,
    :py:class:`~autogluon.eda.analysis.dataset.SpecialTypesAnalysis`, :py:class:`~autogluon.eda.analysis.missing.MissingValuesAnalysis`.
    The components can be present in any combination (assuming their dependencies are satisfied).

    The report requires at least one of the analyses present to be rendered.

    Parameters
    ----------
    headers: bool, default = False
        if `True` then render headers
    namespace: str, default = None
        namespace to use; can be nested like `ns_a.ns_b.ns_c`
     sort_by: Optional[str], default = None
        column to sort the resulting table
     sort_asc: bool, default = True
        if `sort_by` provided, then if sorting should ascending or descending

    Examples
    --------
    >>> import autogluon.eda.analysis as eda
    >>> import autogluon.eda.visualization as viz
    >>> import autogluon.eda.auto as auto
    >>> state = auto.analyze(
    >>>     train_data=..., label=..., return_state=True,
    >>>     anlz_facets=[
    >>>         eda.dataset.DatasetSummary(),
    >>>         eda.dataset.RawTypesAnalysis(),
    >>>         eda.dataset.VariableTypeAnalysis(),
    >>>         eda.dataset.SpecialTypesAnalysis(),
    >>>         eda.missing.MissingValuesAnalysis(),
    >>>     ],
    >>>     viz_facets=[
    >>>         viz.dataset.DatasetStatistics()
    >>>     ]
    >>> )

    See Also
    --------
    :py:class:`~autogluon.eda.analysis.dataset.DatasetSummary`
    :py:class:`~autogluon.eda.analysis.dataset.RawTypesAnalysis`
    :py:class:`~autogluon.eda.analysis.dataset.VariableTypeAnalysis`
    :py:class:`~autogluon.eda.analysis.dataset.SpecialTypesAnalysis`
    :py:class:`~autogluon.eda.analysis.missing.MissingValuesAnalysis`
    """

    def __init__(
        self,
        headers: bool = False,
        namespace: Optional[str] = None,
        sort_by: Optional[str] = None,
        sort_asc: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(namespace, **kwargs)
        self.headers = headers
        self.sort_by = sort_by
        self.sort_asc = sort_asc

    def can_handle(self, state: AnalysisState) -> bool:
        return self.at_least_one_key_must_be_present(
            state, "dataset_stats", "missing_statistics", "raw_type", "special_types"
        )

    def _render(self, state: AnalysisState) -> None:
        datasets = []
        for k in ["dataset_stats", "missing_statistics", "raw_type", "variable_type", "special_types"]:
            if k in state:
                datasets = state[k].keys()
                break

        for ds in datasets:
            # Merge different metrics
            stats = self._merge_analysis_facets(ds, state)
            # Fix counts
            df = pd.DataFrame(stats)
            if "dataset_stats" in state:
                df = self._fix_counts(df, ["unique", "freq"])
            if "missing_statistics" in state:
                df = self._fix_counts(df, ["missing_count"])
            df = df.fillna("")

            self.render_header_if_needed(state, f"`{ds}` dataset summary", ds=ds)
            if self.sort_by in df.columns:
                df = df.sort_values(by=self.sort_by, ascending=self.sort_asc)
            with pd.option_context("display.max_rows", 100 if len(df) <= 100 else 20):
                self.display_obj(df)

    @staticmethod
    def _merge_analysis_facets(ds: str, state: AnalysisState):
        stats: Dict[str, Any] = {}
        if "dataset_stats" in state:
            stats = state.dataset_stats[ds].copy()
        if "missing_statistics" in state:
            stats = {
                **stats,
                **{f"missing_{k}": v for k, v in state.missing_statistics[ds].items() if k in ["count", "ratio"]},
            }
        if "raw_type" in state:
            stats["raw_type"] = state.raw_type[ds]
        if "variable_type" in state:
            stats["variable_type"] = state.variable_type[ds]
        if "special_types" in state:
            stats["special_types"] = state.special_types[ds]
        return stats

    @staticmethod
    def _fix_counts(df: DataFrame, cols: List[str]) -> DataFrame:
        for k in cols:
            if k in df.columns:
                df[k] = df[k].fillna(-1).astype(int).replace({-1: ""})
        return df


class DatasetTypeMismatch(AbstractVisualization, JupyterMixin):
    """
    Display mismatch between raw types between datasets provided. In case if mismatch found, mark the row with a warning.

    The report requires :py:class:`~autogluon.eda.analysis.dataset.RawTypesAnalysis` analysis present.

    Parameters
    ----------
    headers: bool, default = False
        if `True` then render headers
    namespace: str, default = None
        namespace to use; can be nested like `ns_a.ns_b.ns_c`

    Examples
    --------
    >>> import autogluon.eda.analysis as eda
    >>> import autogluon.eda.visualization as viz
    >>> import autogluon.eda.auto as auto
    >>> auto.analyze(
    >>>     train_data=..., test_data=...,
    >>>     anlz_facets=[
    >>>         eda.dataset.RawTypesAnalysis(),
    >>>     ],
    >>>     viz_facets=[
    >>>         viz.dataset.DatasetTypeMismatch()
    >>>     ]
    >>> )

    See Also
    --------
    :py:class:`~autogluon.eda.analysis.dataset.RawTypesAnalysis`
    """

    def __init__(self, headers: bool = False, namespace: Optional[str] = None, **kwargs) -> None:
        super().__init__(namespace, **kwargs)
        self.headers = headers

    def can_handle(self, state: AnalysisState) -> bool:
        return self.all_keys_must_be_present(state, "raw_type")

    def _render(self, state: AnalysisState) -> None:
        df = pd.DataFrame(state.raw_type).sort_index()
        warnings = df.eq(df.iloc[:, 0], axis=0)
        df["warnings"] = warnings.all(axis=1).map({True: "", False: "warning"})
        df.fillna("--", inplace=True)
        df = df[df["warnings"] != ""]

        if len(df) > 0:
            self.render_header_if_needed(state, "Types warnings summary")
            with pd.option_context("display.max_rows", 100 if len(df) <= 100 else 20):
                self.display_obj(df)


class LabelInsightsVisualization(AbstractVisualization, JupyterMixin):
    """
    Render label insights performed by :py:class:`~autogluon.eda.analysis.dataset.LabelInsightsAnalysis`.

    The following insights can be rendered:

    - classification: low cardinality classes detection
    - classification: classes present in test data, but not in the train data
    - regression: out-of-domain labels detection

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

    Parameters
    ----------
    headers: bool, default = False
        if `True` then render headers
    namespace: str, default = None
        namespace to use; can be nested like `ns_a.ns_b.ns_c`

    See Also
    --------
    :py:class:`~autogluon.eda.analysis.dataset.ProblemTypeControl`
    :py:class:`~autogluon.eda.analysis.dataset.LabelInsightsAnalysis`
    """

    def __init__(self, headers: bool = False, namespace: Optional[str] = None, **kwargs) -> None:
        super().__init__(namespace, **kwargs)
        self.headers = headers

    def can_handle(self, state: AnalysisState) -> bool:
        return "label_insights" in state

    def _render(self, state: AnalysisState) -> None:
        insights = state.label_insights

        md_lines: List[str] = []
        self._classification_add_low_cardinality_classes_insights(insights, md_lines)
        self._classification_add_minority_class_imbalance_insights(insights, md_lines)
        self._classification_add_missing_classes_insights(insights, md_lines)
        self._regression_add_out_of_domain_insights(insights, md_lines)

        if len(md_lines) > 0:
            self.render_header_if_needed(state, "Label insights")
            self.render_markdown("\n".join(md_lines))

    @staticmethod
    def _regression_add_out_of_domain_insights(insights: AnalysisState, md_lines: List[str]):
        if insights.ood is not None:
            md_lines.append(
                f" - Rows with out-of-domain labels were found. Consider removing rows with labels outside of this range or expand training data since "
                f"some algorithms (i.e. trees) are unable to extrapolate beyond data present in the training data.\n"
                f"   - `{insights.ood.count}` rows\n"
                f"   - `train_data` values range `{insights.ood.train_range}`\n"
                f"   - `test_data` values range `{insights.ood.test_range}`"
            )

    @staticmethod
    def _classification_add_missing_classes_insights(insights: AnalysisState, md_lines: List[str]):
        if insights.not_present_in_train is not None:
            md_lines.append(
                f" - the following classes are found in `test_data`, but not present in `train_data`: "
                f"`{'`, `'.join(map(str, insights.not_present_in_train))}`. "
                f"Consider either removing the rows with classes not covered or adding more training data covering the classes."
            )

    @staticmethod
    def _classification_add_minority_class_imbalance_insights(insights: AnalysisState, md_lines: List[str]):
        if insights.minority_class_imbalance is not None:
            if insights.minority_class_imbalance.ratio < 0.01:
                severity = "Extreme"
            elif insights.minority_class_imbalance.ratio <= 0.2:
                severity = "Moderate"
            else:
                severity = "Mild"
            md_lines.append(
                f" - {severity} minority class imbalance detected - imbalance ratio is `{insights.minority_class_imbalance.ratio:.2%}`. "
                f"Recommendations:\n"
                f"   - downsample majority class `{insights.minority_class_imbalance.majority_class}` to improve the balance\n"
                f"   - upweight downsampled class so that `sample_weight = original_weight x downsampling_factor`."
                f"[TabularPredictor](https://auto.gluon.ai/stable/api/autogluon.predictor.html#module-0) "
                f"supports this via `sample_weight` parameter"
            )

    @staticmethod
    def _classification_add_low_cardinality_classes_insights(insights: AnalysisState, md_lines: List[str]):
        if insights.low_cardinality_classes is not None:
            classes_info = "\n".join(
                [f"   - class `{k}`: `{v}` instances" for k, v in insights.low_cardinality_classes.instances.items()]
            )
            md_lines.append(
                f" - Low-cardinality classes are detected. It is recommended to have at least `{insights.low_cardinality_classes.threshold}` "
                f"instances per class. Consider adding more data to cover the classes or remove such rows.\n"
                f"{classes_info}"
            )
