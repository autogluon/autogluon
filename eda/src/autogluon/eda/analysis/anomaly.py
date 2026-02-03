from __future__ import print_function

import builtins as __builtin__
import contextlib
import logging
from functools import partial
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from pyod.models.base import BaseDetector
from pyod.models.copod import COPOD
from pyod.models.iforest import IForest
from pyod.models.lof import LOF
from pyod.models.suod import SUOD

from autogluon.common.utils.cv_splitter import CVSplitter
from autogluon.common.utils.resource_utils import ResourceManager

from .. import AnalysisState
from .base import AbstractAnalysis

__all__ = ["AnomalyDetector", "AnomalyDetectorAnalysis"]

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def _suod_silent_print(silent=True):  # pragma: no cover
    """
    Workaround to suppress log clutter from SUOD

    See Also
    --------
    https://github.com/yzhao062/SUOD/pull/7
    https://github.com/yzhao062/SUOD/pull/12
    """

    orig_fn = joblib.Parallel._print
    orig_print = __builtin__.print

    def silent_print(*args, **kwargs):
        return

    def _silent_print(*args, **kwargs):
        if args != () and kwargs != {}:
            orig_print(*args, **kwargs)

    if silent:
        joblib.Parallel._print = silent_print
        __builtin__.print = _silent_print  # type: ignore[assignment]
    try:
        yield
    finally:
        if silent:
            joblib.Parallel._print = orig_fn
            __builtin__.print = orig_print


class AnomalyDetector:
    """
    Wrapper for anomaly detector algorithms.

    :py:meth:`~autogluon.eda.analysis.anomaly.AnomalyDetector.fit_transform` automatically creates
    cross-validation splits and fits detectors on each of them. The scores produced for the training
    data are produced using out-of-folds predictions

    :py:meth:`~autogluon.eda.analysis.anomaly.AnomalyDetector.transform` uses average of scores from
    the detectors trained on the folds.

    Please note: the data passed into the fit/transform must be already pre-processed;
    numeric columns must have no NaNs.

    Parameters
    ----------
    label: str
        dataset's label column name
    n_folds: int, default = 5,
        number of folds to use when training detectors
    detector_list: Optional[List[BaseDetector]], default = None
        list of detectors to ensemble. If `None`, then use the standard list:
         - LOF(n_neighbors=15)
         - LOF(n_neighbors=20)
         - LOF(n_neighbors=25)
         - LOF(n_neighbors=35)
         - COPOD
         - IForest(n_estimators=100)
         - IForest(n_estimators=200)
        See `pyod <https://pyod.readthedocs.io/en/latest/pyod.models.html>`_ documentation for the full model list.
    silent: bool, default = True
        Suppress SUOD logs if `True`
    detector_kwargs
        kwargs to pass into detector
    """

    def __init__(
        self,
        label: str,
        n_folds: int = 5,
        detector_list: Optional[List[BaseDetector]] = None,
        silent: bool = True,
        **detector_kwargs,
    ) -> None:
        self.label = label
        self.n_folds = n_folds
        self.silent = silent
        if detector_list is None:
            detector_list = AnomalyDetector._get_default_detector_list()
        self.detector_list = detector_list

        # Can't go beyond 4 - SUOD is throwing errors
        num_cpus = min(ResourceManager.get_cpu_count(), 4)

        # Don't use `bps_flag=True` - it's using pre-trained models which aren't loading in newer versions of sklearn
        suod_defaults = dict(
            base_estimators=self.detector_list, n_jobs=num_cpus, combination="average", bps_flag=False, verbose=False
        )
        self._suod_kwargs = {**suod_defaults, **detector_kwargs}
        self._detectors: Optional[List[BaseDetector]] = None
        self._train_index_to_detector: Optional[Dict[int, Any]] = None
        self.original_features: Optional[List[str]] = None

    @staticmethod
    def _get_default_detector_list():
        return [
            LOF(n_neighbors=15),
            LOF(n_neighbors=20),
            LOF(n_neighbors=25),
            LOF(n_neighbors=35),
            COPOD(),
            IForest(n_estimators=100),
            IForest(n_estimators=200),
        ]

    @property
    def problem_type(self):
        return "regression"

    def fit_transform(self, train_data: pd.DataFrame) -> pd.Series:
        """
        Automatically creates cross-validation splits and fits detectors on each of them.
        The scores produced for the training data are produced using out-of-folds predictions

        Parameters
        ----------
        train_data: pd.DataFrame
            training data; must be already pre-processed; numeric columns must have NaNs filled

        Returns
        -------
        out-of-folds anomaly scores for the training data

        """
        self._detectors = []
        self._train_index_to_detector = {}
        splitter = CVSplitter(n_splits=self.n_folds)
        x, y = train_data.drop(columns=self.label), train_data[self.label]
        self.original_features = x.columns

        folds_scores = []
        for i, (train_idx, val_idx) in enumerate(splitter.split(x, y)):
            x_train = x.iloc[train_idx]
            x_val = x.iloc[val_idx]

            with _suod_silent_print(self.silent):
                detector = SUOD(**self._suod_kwargs)
                np.int = int  # type: ignore[attr-defined] # workaround to address shap's use of old numpy APIs
                self._detectors.append(detector.fit(x_train))
                self._train_index_to_detector = {**self._train_index_to_detector, **{idx: i for idx in x_train.index}}
                val_scores = detector.decision_function(x_val)  # outlier scores
            folds_scores.append(pd.Series(name="score", data=val_scores, index=x_val.index))
        return pd.concat(folds_scores, axis=0)[x.index]

    def transform(self, x: pd.DataFrame):
        """
        Predict anomaly scores for the provided inputs.
        This method uses average of scores produced by all the detectors trained on folds.

        Parameters
        ----------
        x: pd.DataFrame
            data to score; must be already pre-processed; numeric columns must have NaNs filled

        Returns
        -------
        anomaly scores for the passed data
        """
        assert self._detectors is not None, "Detector is not fit - call `fit_transform` before calling `transform`"

        folds_scores = []
        for detector in self._detectors:
            with _suod_silent_print(self.silent):
                y_test_scores = detector.decision_function(x[self.original_features])
            folds_scores.append(pd.DataFrame({"score": y_test_scores}, index=x.index))
        score = pd.concat([df.score for df in folds_scores], axis=1).mean(axis=1)
        score.name = "score"

        return score[x.index]

    def predict(self, x):
        """
        API-compatibility wrapper for :py:meth:`~autogluon.eda.analysis.anomaly.AnomalyDetector.transform`
        """
        return self.transform(x)


class AnomalyDetectorAnalysis(AbstractAnalysis):
    """
    Anomaly detection analysis.

    The analysis automatically creates cross-validation splits and fits detectors on each of them using
    `train_data` input. The scores produced for the training data are produced using out-of-folds predictions.
    All other datasets scores are produced using average of scores from detectors trained on individual folds (bag).

    Please note, the analysis expects the data is ready to for fitting; all numeric columns must not have NaNs.
    Pre-processing can be performed using :py:class:`~autogluon.eda.analysis.transform.ApplyFeatureGenerator`
    and :py:class:`~autogluon.eda.analysis.dataset.ProblemTypeControl` (see example for more details).

    State attributes

    - `anomaly_detection.scores.<dataset>`
        scores for each of the datasets passed into analysis (i.e. `train_data`, `test_data`)
    - `anomaly_detection.explain_rows_fns.<dataset>`
        if `store_explainability_data=True`, then analysis will store helper functions into this
        variable. The function can be used later via :py:meth:`~autogluon.eda.auto.simple.explain_rows`
        and automatically pre-populates `train_data`, `model` and `rows` parameters when called
        (see example for more details)


    Parameters
    ----------
    n_folds: int, default = 5
        number of folds to use when training detectors; default is 5 folds.
    store_explainability_data: bool, default = False
        if `True` analysis will store helper functions into this variable.
        The function can be used later via :py:meth:`~autogluon.eda.auto.simple.explain_rows`
        and automatically pre-populates `train_data`, `model` and `rows` parameters when called
        (see example for more details)
    parent: Optional[AbstractAnalysis], default = None
        parent Analysis
    children: List[AbstractAnalysis], default = []
        wrapped analyses; these will receive sampled `args` during `fit` call
    state: Optional[AnalysisState], default = None
        state to be updated by this fit function
    anomaly_detector_kwargs
        kwargs for :py:class:`~autogluon.eda.analysis.anomaly.AnomalyDetector`


    See Also
    --------
    :py:class:`~autogluon.eda.analysis.anomaly.AnomalyDetector`
    :py:class:`~autogluon.eda.visualization.anomaly.AnomalyScoresVisualization`
    :py:class:`~autogluon.eda.analysis.transform.ApplyFeatureGenerator`
    :py:class:`~autogluon.eda.analysis.dataset.ProblemTypeControl`

    """

    def __init__(
        self,
        n_folds: int = 5,
        store_explainability_data: bool = False,
        parent: Optional[AbstractAnalysis] = None,
        children: Optional[List[AbstractAnalysis]] = None,
        state: Optional[AnalysisState] = None,
        **anomaly_detector_kwargs,
    ) -> None:
        super().__init__(parent, children, state, **anomaly_detector_kwargs)
        self.n_folds = n_folds
        self.store_explainability_data = store_explainability_data

    def can_handle(self, state: AnalysisState, args: AnalysisState) -> bool:
        args_present = self.all_keys_must_be_present(args, "train_data", "label")
        no_nans = True
        if args_present:
            for ds, df in self.available_datasets(args):
                cols_with_nas = [c for c in df.columns if df[c].dtype != "object" and df[c].hasnans]
                if len(cols_with_nas) > 0:
                    self.logger.warning(
                        f"{ds}: NaNs are present in the following columns: {cols_with_nas};"
                        f" please fill them before calling this method."
                    )
                    no_nans = False

        return args_present and no_nans

    def _fit(self, state: AnalysisState, args: AnalysisState, **fit_kwargs) -> None:
        det = self._create_detector(args)
        scores = det.fit_transform(args.train_data)
        s = {"scores": {"train_data": scores}}
        if self.store_explainability_data:
            s["explain_rows_fns"] = {
                "train_data": partial(AnomalyDetectorAnalysis.explain_rows_fn, args, det, "train_data")
            }

        for ds, df in self.available_datasets(args):
            if ds == "train_data":
                continue
            s["scores"][ds] = det.transform(df)
            if self.store_explainability_data:
                s["explain_rows_fns"][ds] = partial(AnomalyDetectorAnalysis.explain_rows_fn, args, det, ds)

        state["anomaly_detection"] = s

    def _create_detector(self, args) -> AnomalyDetector:
        return AnomalyDetector(label=args.label, n_folds=self.n_folds, **self.args)

    @staticmethod
    def explain_rows_fn(args: AnalysisState, detector: AnomalyDetector, dataset: str, dataset_row_ids: List[Any]):
        """
        Prepares arguments for :py:meth:`~autogluon.eda.auto.simple.explain_rows` call to explain anomaly scores contributions

        Parameters
        ----------
        args: AnalysisState,
            args from the analysis call (will be pre-populated)
        detector: AnomalyDetector
            detector to use for the prediction (will be pre-populated)
        dataset: str
            dataset to use (will be pre-populated)
        dataset_row_ids: List[any]
            list of row ids to explain from the specified `dataset`

        Returns
        -------
        Dict of arguments to pass into

        See Also
        --------
        :py:meth:`~autogluon.eda.auto.simple.explain_rows`


        """
        missing_ids = [item for item in dataset_row_ids if item not in args[dataset].index]
        assert len(missing_ids) == 0, f"The following ids are missing in `{dataset_row_ids}`: {missing_ids}"
        logger.info(
            "Please note that the feature values shown on the charts are transformed into an internal representation; "
            "they may be encoded or modified based on internal preprocessing. Refer to the original datasets for the actual feature values."
        )
        if dataset == "train_data":
            logger.warning(
                "Warning: The `train_data` dataset is used for explanation. The detector has seen the data, and estimates may be overly optimistic. "
                "Although the anomaly score in the explanation might not match, the magnitude of the features can still be utilized to "
                "evaluate the impact of the feature on the anomaly score."
            )
        return dict(
            train_data=args.train_data,
            model=detector,
            rows=args[dataset].loc[dataset_row_ids],
        )
