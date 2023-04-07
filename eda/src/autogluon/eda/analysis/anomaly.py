import builtins as __builtin__
import contextlib
from typing import List, Optional

import joblib
import pandas as pd
from pyod.models.base import BaseDetector
from pyod.models.copod import COPOD
from pyod.models.iforest import IForest
from pyod.models.lof import LOF
from pyod.models.suod import SUOD

from autogluon.common.utils.resource_utils import ResourceManager
from autogluon.core.utils import CVSplitter

from .. import AnalysisState
from .base import AbstractAnalysis

__all__ = ["AnomalyDetector", "AnomalyDetectorAnalysis"]


@contextlib.contextmanager
def _suod_silent_print(silent=True):
    """
    Workaround to suppress log clutter

    See Also
    --------
    - https://github.com/yzhao062/SUOD/pull/7
    - https://github.com/yzhao062/SUOD/pull/12
    """
    orig_fn = joblib.Parallel._print
    orig_print = __builtin__.print

    def silent_print(self, msg, msg_args):
        return

    @staticmethod
    def _silent_print():
        pass

    if silent:
        joblib.Parallel._print = silent_print
        __builtin__.print = _silent_print
    try:
        yield
    finally:
        if silent:
            joblib.Parallel._print = orig_fn
            __builtin__.print = orig_print


class AnomalyDetector:
    def __init__(
        self,
        label: str,
        n_folds: int = 5,
        detector_list: Optional[List[BaseDetector]] = None,
        silent: bool = True,
        **suod_kwargs,
    ) -> None:
        self.label = label
        self.n_folds = n_folds
        self.silent = silent
        if detector_list is None:
            detector_list = [
                LOF(n_neighbors=15),
                LOF(n_neighbors=20),
                LOF(n_neighbors=25),
                LOF(n_neighbors=35),
                COPOD(),
                IForest(n_estimators=100),
                IForest(n_estimators=200),
            ]
        self.detector_list = detector_list

        # Can't go beyond 4 - SUOD is throwing errors
        num_cpus = min(ResourceManager.get_cpu_count(), 4)

        suod_defaults = dict(base_estimators=self.detector_list, n_jobs=num_cpus, combination="average", verbose=False)
        self.suod_kwargs = {**suod_defaults, **suod_kwargs}
        self.detectors = None
        self.original_features = None
        self._train_index_to_detector = None

    @property
    def problem_type(self):
        return "regression"

    def fit_transform(self, train_data: pd.DataFrame) -> pd.Series:
        self.detectors = []
        self._train_index_to_detector = {}
        splitter = CVSplitter(n_splits=self.n_folds)
        x, y = train_data.drop(columns=self.label), train_data[self.label]
        self.original_features = x.columns

        folds_scores = []
        for i, (train_idx, val_idx) in enumerate(splitter.split(x, y)):
            x_train = x.iloc[train_idx]
            x_val = x.iloc[val_idx]

            with _suod_silent_print(self.silent):
                detector = SUOD(**self.suod_kwargs)
                self.detectors.append(detector.fit(x_train))
                self._train_index_to_detector = {**self._train_index_to_detector, **{idx: i for idx in x_train.index}}
                val_scores = detector.decision_function(x_val)  # outlier scores
            folds_scores.append(pd.Series(name="score", data=val_scores, index=x_val.index))
        return pd.concat(folds_scores, axis=0)[x.index]

    def transform(self, x):
        folds_scores = []
        for detector in self.detectors:
            with _suod_silent_print(self.silent):
                y_test_scores = detector.decision_function(x[self.original_features])
            folds_scores.append(pd.DataFrame({"score": y_test_scores}, index=x.index))
        score = pd.concat([df.score for df in folds_scores], axis=1).mean(axis=1)
        score.name = "score"

        return score[x.index]

    def predict(self, x):
        return self.transform(x)


class AnomalyDetectorAnalysis(AbstractAnalysis):
    def __init__(
        self,
        n_folds: int = 5,
        parent: Optional[AbstractAnalysis] = None,
        children: Optional[List[AbstractAnalysis]] = None,
        state: Optional[AnalysisState] = None,
        store_explainability_data: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(parent, children, state, **kwargs)
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
        det = AnomalyDetector(label=args.label, n_folds=self.n_folds)
        scores = det.fit_transform(args.train_data)
        s = {"scores": {"train_data": scores}}
        if self.store_explainability_data:
            s["detector"] = det
            s["transformed_features"] = {"train_data": args.train_data}

        for ds, df in self.available_datasets(args):
            if ds == "train_data":
                continue
            s["scores"][ds] = det.transform(df)
            if self.store_explainability_data:
                s["transformed_features"][ds] = df

        state["anomaly_detection"] = s
