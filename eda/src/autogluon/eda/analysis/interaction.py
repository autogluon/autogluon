import warnings
from typing import List, Optional, Dict, Any, Union

import numpy as np
import pandas as pd
import phik  # noqa - required for significance_matrix instrumentation on pandas dataframes
from pandas.core.dtypes.common import is_numeric_dtype
from scipy import stats
from scipy.cluster import hierarchy as hc
from scipy.stats import spearmanr

from .base import AbstractAnalysis
from .. import AnalysisState

__all__ = ["Correlation", "CorrelationSignificance", "FeatureInteraction", "DistributionFit"]


class Correlation(AbstractAnalysis):
    """
    Correlation analysis.

    Parameters
    ----------
    method: str  {'pearson', 'kendall', 'spearman', 'phik'}, default='spearman'
        Method of correlation:
            * pearson : standard correlation coefficient
            * kendall : Kendall Tau correlation coefficient
            * spearman : Spearman rank correlation
            * phik : phi_k correlation
                Correlation matrix of bivariate gaussian derived from chi2-value Chi2-value
                gets converted into correlation coefficient of bivariate gauss with correlation
                value rho, assuming given binning and number of records. Correlation coefficient
                value is between 0 and 1. Bivariate gaussian's range is set to [-5,5] by construction.
                See Also `phik <https://github.com/KaveIO/PhiK>`_ documentation.

    focus_field: Optional[str], default = None
        field name to focus. Specifying a field would filter all correlations only when they are >= `focus_field_threshold`
        This is helpful when dealing with a large number of variables.
    focus_field_threshold: float, default = 0.5
        a cut-off threshold when `focus_field` is specified
    parent: Optional[AbstractAnalysis], default = None
        parent Analysis
    children: List[AbstractAnalysis], default = []
        wrapped analyses; these will receive sampled `args` during `fit` call

    See Also
    --------
    `phik <https://github.com/KaveIO/PhiK>`_ documentation

    """

    def __init__(
        self,
        method: str = "spearman",
        focus_field: Optional[str] = None,
        focus_field_threshold: float = 0.5,
        parent: Optional[AbstractAnalysis] = None,
        children: Optional[List[AbstractAnalysis]] = None,
        **kwargs,
    ) -> None:
        super().__init__(parent, children, **kwargs)
        assert method in ["pearson", "kendall", "spearman", "phik"]
        self.method = method
        self.focus_field = focus_field
        self.focus_field_threshold = focus_field_threshold

    def can_handle(self, state: AnalysisState, args: AnalysisState) -> bool:
        return True

    def _fit(self, state: AnalysisState, args: AnalysisState, **fit_kwargs) -> None:
        state.correlations = {}
        state.correlations_method = self.method
        for (ds, df) in self.available_datasets(args):
            if self.method == "phik":
                state.correlations[ds] = df.phik_matrix(**self.args, verbose=False)
            else:
                state.correlations[ds] = df.corr(method=self.method, **self.args)

            if self.focus_field is not None and self.focus_field in state.correlations[ds].columns:
                state.correlations_focus_field = self.focus_field
                state.correlations_focus_field_threshold = self.focus_field_threshold
                state.correlations_focus_high_corr = {}
                df_corr = state.correlations[ds]
                df_corr = df_corr[df_corr[self.focus_field].abs() >= self.focus_field_threshold]
                keep_cols = df_corr.index.values
                state.correlations[ds] = df_corr[keep_cols]

                high_corr = (
                    state.correlations[ds][[self.focus_field]]
                    .sort_values(self.focus_field, ascending=False)
                    .drop(self.focus_field)
                )
                state.correlations_focus_high_corr[ds] = high_corr


class CorrelationSignificance(AbstractAnalysis):
    """
    Significance of correlation of all variable combinations in the DataFrame.
    See :py:meth:`~phik.significance.significance_matrix` for more details.
    This analysis requires :py:class:`~autogluon.eda.analysis.interaction.Correlation` results to be
    available in the state.

    See Also
    --------
    `phik <https://github.com/KaveIO/PhiK>`_ documentation
    :py:meth:`~phik.significance.significance_matrix`
    :py:class:`~autogluon.eda.analysis.interaction.Correlation`
    """

    def can_handle(self, state: AnalysisState, args: AnalysisState) -> bool:
        return self.all_keys_must_be_present(state, "correlations", "correlations_method")

    def _fit(self, state: AnalysisState, args: AnalysisState, **fit_kwargs) -> None:
        state.significance_matrix = {}
        for (ds, df) in self.available_datasets(args):
            state.significance_matrix[ds] = df[state.correlations[ds].columns].significance_matrix(
                **self.args, verbose=False
            )


class FeatureInteraction(AbstractAnalysis):
    """
    Feature interaction analysis

    Parameters
    ----------
    x: Optional[str], default = None
        variable to analyse which would be placed on x-axis
    y: Optional[str], default = None
        variable to analyse which would be placed on y-axis
    hue: Optional[str], default = None
        variable to use as hue in x/y-analysis.
    key: Optional[str], default = None
        key to use to store the analysis in the state; the value is later to be used by FeatureInteractionVisualization.
        If the key is not provided, then use one of theform: 'x:A|y:B|hue:C' (omit corresponding x/y/hue if the value not provided)
        See also :py:class:`~autogluon.eda.visualization.interaction.FeatureInteractionVisualization`
    parent: Optional[AbstractAnalysis], default = None
        parent Analysis
    children: Optional[List[AbstractAnalysis]], default None
        wrapped analyses; these will receive sampled `args` during `fit` call
    kwargs

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> import autogluon.eda.analysis as eda
    >>> import autogluon.eda.visualization as viz
    >>> import autogluon.eda.auto as auto
    >>>
    >>> df_train = pd.DataFrame(...)
    >>>
    >>> state = auto.analyze(
    >>>     train_data=df_train, label='Survived',
    >>>     anlz_facets=[
    >>>         eda.dataset.RawTypesAnalysis(),
    >>>         eda.interaction.FeatureInteraction(key='target_col', x='Survived'),
    >>>         eda.interaction.FeatureInteraction(key='target_col_vs_age', x='Survived', y='Age')
    >>>     ],
    >>>     viz_facets=[
    >>>         # Bar Plot with counts per each of the values in Survived
    >>>         viz.interaction.FeatureInteractionVisualization(key='target_col', headers=True),
    >>>         # Box Plot Survived vs Age
    >>>         viz.interaction.FeatureInteractionVisualization(key='target_col_vs_age', headers=True),
    >>>     ]
    >>> )
    >>>
    >>> # Simplified shortcut for interactions: scatter plot of Fare vs Age colored based on Survived values.
    >>> auto.analyze_interaction(x='Fare', y='Age', hue='Survived', train_data=df_train)
    """

    def __init__(
        self,
        x: Optional[str] = None,
        y: Optional[str] = None,
        hue: Optional[str] = None,
        key: Optional[str] = None,
        parent: Optional[AbstractAnalysis] = None,
        children: Optional[List[AbstractAnalysis]] = None,
        **kwargs,
    ) -> None:
        super().__init__(parent, children, **kwargs)
        self.x = x
        self.y = y
        self.hue = hue
        self.key = key

    def can_handle(self, state: AnalysisState, args: AnalysisState) -> bool:
        return self.all_keys_must_be_present(state, "raw_type")

    def _fit(self, state: AnalysisState, args: AnalysisState, **fit_kwargs):
        cols = {
            "x": self.x,
            "y": self.y,
            "hue": self.hue,
        }

        self.key = self._generate_key_if_not_provided(self.key, cols)

        cols = {k: v for k, v in cols.items() if v is not None}

        interactions: Dict[str, Dict[str, Any]] = state.get("interactions", {})
        for (ds, df) in self.available_datasets(args):
            missing_cols = [c for c in cols.values() if c not in df.columns]
            if len(missing_cols) == 0:
                df = df[cols.values()]
                interaction = {
                    "features": cols,
                    "data": df,
                }
                if ds not in interactions:
                    interactions[ds] = {}
                interactions[ds][self.key] = interaction
        state.interactions = interactions

    def _generate_key_if_not_provided(self, key: Optional[str], cols: Dict[str, Optional[str]]) -> str:
        # if key is not provided, then convert to form: 'x:A|y:B|hue:C'; if values is not provided, then skip the value
        if key is None:
            key_parts = []
            for k, v in cols.items():
                if v is not None:
                    key_parts.append(f"{k}:{v}")
            key = "|".join(key_parts)
        return key


class DistributionFit(AbstractAnalysis):
    """
    This component attempts to fit various distributions for further plotting via
    :py:class:`~autogluon.eda.visualization.interaction.FeatureInteractionVisualization`.

    The data specified in `columns` must be numeric to be considered for fitting (categorical variables are not supported).

    Only the distributions with statistical significance above `pvalue_min` threshold will be included in the results.

    Note: this analysis is an augmentation for :py:class:`~autogluon.eda.analysis.interaction.FeatureInteraction` and should be used in pair
    to be visualized via :py:class:`~autogluon.eda.visualization.interaction.FeatureInteractionVisualization`.

    Parameters
    ----------
    columns: Union[str, List[str]]
        colums to be included into analysis. Can be passed as a string or a list of strings.
    pvalue_min: float = 0.01,
        min pvalue to consider including distribution fit in the results.
    keep_top_n: Optional[int] = None,
        how many distributions exceeding `pvalue_min` to include in the results. I.e. if `keep_top_n=3`,
        but 10 distributions satisfied `pvalue_min`, only top 3 will be included.
        If not specified and `distributions_to_fit` is not provided, then only top 3 will be included in the results.
    distributions_to_fit: Optional[Union[str, List[str]]] = None,
        list of distributions to fit. See `DistributionFit.AVAILABLE_DISTRIBUTIONS` for the list of supported values.
        See `scipy <https://docs.scipy.org/doc/scipy/reference/stats.html>`_ documentation for each distribution details.
        If not specified, then all supported distributions will be attempted to fit.
    parent: Optional[AbstractAnalysis], default = None
        parent Analysis
    children: Optional[List[AbstractAnalysis]], default None
        wrapped analyses; these will receive sampled `args` during `fit` call
    kwargs

    Examples
    --------
    >>> import autogluon.eda.analysis as eda
    >>> import autogluon.eda.visualization as viz
    >>> import autogluon.eda.auto as auto
    >>> import pandas as pd
    >>> import numpy as np
    >>>
    >>> df_train = pd.DataFrame(...)
    >>>
    >>> auto.analyze(
    >>>     train_data=df_train, label=target_col,
    >>>     anlz_facets=[
    >>>         eda.dataset.RawTypesAnalysis(),
    >>>         eda.interaction.DistributionFit(columns=['Fare', 'Age'], distributions_to_fit=['lognorm', 'beta', 'gamma', 'fisk']),
    >>>         eda.interaction.FeatureInteraction(key='age-chart', x='Age'),
    >>>
    >>>     ],
    >>>     viz_facets=[
    >>>         viz.interaction.FeatureInteractionVisualization(key='age-chart', headers=True),
    >>>     ]
    >>> )

    See Also
    --------
    :py:class:`~autogluon.eda.analysis.interaction.FeatureInteraction`
    :py:class:`~autogluon.eda.visualization.interaction.FeatureInteractionVisualization`
    `scipy <https://docs.scipy.org/doc/scipy/reference/stats.html>`_ documentation for each distribution details

    """

    # Getting the list of distributions: https://docs.scipy.org/doc/scipy/tutorial/stats.html#getting-help
    AVAILABLE_DISTRIBUTIONS = sorted([d for d in dir(stats) if isinstance(getattr(stats, d), stats.rv_continuous)])

    def __init__(
        self,
        columns: Union[str, List[str]],
        pvalue_min: float = 0.01,
        keep_top_n: Optional[int] = None,
        distributions_to_fit: Optional[Union[str, List[str]]] = None,
        parent: Optional[AbstractAnalysis] = None,
        children: Optional[List[AbstractAnalysis]] = None,
        **kwargs,
    ) -> None:
        super().__init__(parent, children, **kwargs)

        if keep_top_n is None and distributions_to_fit is None:
            keep_top_n = 3

        if isinstance(columns, str):
            columns = [columns]
        self.columns = columns

        self.pvalue_min = pvalue_min
        self.keep_top_n = keep_top_n

        if distributions_to_fit is None:
            distributions_to_fit = self.AVAILABLE_DISTRIBUTIONS
        if isinstance(distributions_to_fit, str):
            distributions_to_fit = [distributions_to_fit]
        not_supported = [d for d in distributions_to_fit if d not in self.AVAILABLE_DISTRIBUTIONS]
        if len(not_supported) > 0:
            raise ValueError(
                f"The following distributions are not supported: {sorted(not_supported)}. "
                f"Supported distributions are {sorted(self.AVAILABLE_DISTRIBUTIONS)}"
            )
        self.distributions_to_fit = distributions_to_fit

    def can_handle(self, state: AnalysisState, args: AnalysisState) -> bool:
        return True

    def _fit(self, state: AnalysisState, args: AnalysisState, **fit_kwargs) -> None:
        state.distributions_fit = {}
        for (ds, df) in self.available_datasets(args):
            state.distributions_fit[ds] = {}
            for c in self.columns:
                if c in df.columns:
                    col = df[c]
                    col = col[col.notna()]  # skip NaNs
                    dist = self._fit_dist(col, self.pvalue_min)
                    if dist is not None:
                        state.distributions_fit[ds][c] = dist

    def _fit_dist(self, series, pvalue_min=0.01):
        results = {}
        if not is_numeric_dtype(series):
            self.logger.warning(f"{series.name}: distribution cannot be fit; only numeric columns are supported")
            return None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for i in self.distributions_to_fit:
                dist = getattr(stats, i)
                param = dist.fit(series)
                statistic, pvalue = stats.kstest(series, i, args=param)
                if pvalue >= pvalue_min:
                    results[i] = {
                        "param": param,
                        "statistic": statistic,
                        "pvalue": pvalue,
                    }
            if len(results) == 0:
                self.logger.warning(
                    f"{series.name}: none of the distributions were able to fit to satisfy specified pvalue_min: {self.pvalue_min}"
                )
                return None
            df = pd.DataFrame(results).T.sort_values("pvalue", ascending=False)
            if self.keep_top_n is not None:
                df = df[: self.keep_top_n]
            results = df.T.to_dict()
            return results


class FeatureDistanceAnalysis(AbstractAnalysis):
    """
    The component performs feature correlation distance analysis using Spearman rank correlation and hierarchical clustering
    for the data passed in `train_data` excluding `label`.
    The near duplicates grouping is automatically suggested given `near_duplicates_threshold`.
    The results can be visualized using :py:class:`~autogluon.eda.visualization.interaction.FeatureDistanceAnalysisVisualization`.

    Note: it is recommended to apply :py:class:`~autogluon.eda.analysis.transform.ApplyFeatureGenerator` before the analysis
    to ensure correlations are calculated for categorical variables.

    Parameters
    ----------
    near_duplicates_threshold: float, default = 0.01
        defines feature distance to be considered as near duplicates
    parent: Optional[AbstractAnalysis], default = None
        parent Analysis
    children: Optional[List[AbstractAnalysis]], default None
        wrapped analyses; these will receive sampled `args` during `fit` call
    kwargs

    Examples
    --------
    >>> import autogluon.eda.analysis as eda
    >>> import autogluon.eda.visualization as viz
    >>> import autogluon.eda.auto as auto
    >>> import pandas as pd
    >>> import numpy as np
    >>>
    >>> df_train = pd.DataFrame(...)
    >>>
    >>> auto.analyze(
    >>>     train_data=df_train, label=target_col,
    >>>     anlz_facets=[
    >>>         eda.transform.ApplyFeatureGenerator(category_to_numbers=True, children=[
    >>>             eda.interaction.FeatureDistanceAnalysis(near_duplicates_threshold=0.7),
    >>>         ])
    >>>     ],
    >>>     viz_facets=[
    >>>         viz.interaction.FeatureDistanceAnalysisVisualization(fig_args=dict(figsize=(12,6))),
    >>>     ]
    >>> )

    See Also
    --------
    :py:class:`~autogluon.eda.analysis.transform.ApplyFeatureGenerator`
    :py:class:`~autogluon.eda.visualization.interaction.FeatureDistanceAnalysisVisualization`
    `Removing redundant features <https://github.com/fastai/book_nbs/blob/master/10_tabular.ipynb>`_ section of
        Jeremy Howard's "Deep Learning for Coders with Fastai and PyTorch" book.

    """

    def __init__(
        self,
        near_duplicates_threshold: float = 0.01,
        parent: Optional[AbstractAnalysis] = None,
        children: Optional[List[AbstractAnalysis]] = None,
        **kwargs,
    ) -> None:
        super().__init__(parent, children, **kwargs)
        self.near_duplicates_threshold = near_duplicates_threshold

    def can_handle(self, state: AnalysisState, args: AnalysisState) -> bool:
        return self.all_keys_must_be_present(args, "train_data", "label", "feature_generator")

    def _fit(self, state: AnalysisState, args: AnalysisState, **fit_kwargs) -> None:
        x = args.train_data.drop(labels=[args.label], axis=1)
        corr = np.round(spearmanr(x).correlation, 4)
        np.fill_diagonal(corr, 1)
        corr_condensed = hc.distance.squareform(1 - np.nan_to_num(corr))
        z = hc.linkage(corr_condensed, method="average")
        columns = list(x.columns)
        s = {
            "columns": columns,
            "linkage": z,
            "near_duplicates_threshold": self.near_duplicates_threshold,
            "near_duplicates": self.__get_linkage_clusters(z, columns, self.near_duplicates_threshold),
        }
        state["feature_distance"] = s

    @staticmethod
    def __get_linkage_clusters(linkage, columns, threshold: float):
        idx_to_col = {i: v for i, v in enumerate(columns)}
        idx_to_dist: Dict[int, float] = {}
        clusters: Dict[int, List[int]] = {}
        for (f1, f2, d, _l), i in zip(linkage, np.arange(len(idx_to_col), len(idx_to_col) + len(linkage))):
            idx_to_dist[i] = d
            f1 = int(f1)
            f2 = int(f2)
            if d <= threshold:
                clusters[i] = [*clusters.pop(f1, [f1]), *clusters.pop(f2, [f2])]

        results = []
        for i, nodes in clusters.items():
            d = idx_to_dist[i]
            nodes = [idx_to_col[n] for n in nodes]
            results.append(
                {
                    "nodes": sorted(nodes),
                    "distance": d,
                }
            )

        return results
