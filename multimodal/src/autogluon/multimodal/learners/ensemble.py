import copy
import json
import logging
import os
import pathlib
import pickle
import pprint
import time
import warnings
from typing import Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from autogluon.core.metrics import Scorer
from autogluon.core.models.greedy_ensemble.ensemble_selection import EnsembleSelection

from .. import version as ag_version
from ..constants import BINARY, LOGITS, MULTICLASS, REGRESSION, TEST, VAL, Y_PRED, Y_TRUE
from ..optim import compute_score
from ..utils import (
    extract_from_output,
    get_dir_ckpt_paths,
    logits_to_prob,
    on_fit_end_message,
    update_ensemble_hyperparameters,
)
from .base import BaseLearner

logger = logging.getLogger(__name__)


class EnsembleLearner(BaseLearner):
    def __init__(
        self,
        label: Optional[str] = None,
        problem_type: Optional[str] = None,
        presets: Optional[str] = "high_quality",
        eval_metric: Optional[Union[str, Scorer]] = None,
        hyperparameters: Optional[dict] = None,
        path: Optional[str] = None,
        verbosity: Optional[int] = 2,
        warn_if_exist: Optional[bool] = True,
        enable_progress_bar: Optional[bool] = None,
        ensemble_size: Optional[int] = 2,
        ensemble_mode: Optional[str] = "one_shot",
        **kwargs,
    ):
        """
        Parameters
        ----------
        label
            Name of the column that contains the target variable to predict.
        problem_type
            Type of the prediction problem. We support standard problems like

            - 'binary': Binary classification
            - 'multiclass': Multi-class classification
            - 'regression': Regression
            - 'classification': Classification problems include 'binary' and 'multiclass' classification.
        presets
            Presets regarding model quality, e.g., best_quality, high_quality, and medium_quality.
        eval_metric
            Evaluation metric name. If `eval_metric = None`, it is automatically chosen based on `problem_type`.
            Defaults to 'accuracy' for multiclass classification, `roc_auc` for binary classification, and 'root_mean_squared_error' for regression.
        hyperparameters
            This is to override some default configurations.
            For example, changing the text and image backbones can be done by formatting:

            a string
            hyperparameters = "model.hf_text.checkpoint_name=google/electra-small-discriminator model.timm_image.checkpoint_name=swin_small_patch4_window7_224"

            or a list of strings
            hyperparameters = ["model.hf_text.checkpoint_name=google/electra-small-discriminator", "model.timm_image.checkpoint_name=swin_small_patch4_window7_224"]

            or a dictionary
            hyperparameters = {
                            "model.hf_text.checkpoint_name": "google/electra-small-discriminator",
                            "model.timm_image.checkpoint_name": "swin_small_patch4_window7_224",
                        }
        path
            Path to directory where models and intermediate outputs should be saved.
            If unspecified, a time-stamped folder called "AutogluonAutoMM/ag-[TIMESTAMP]"
            will be created in the working directory to store all models.
            Note: To call `fit()` twice and save all results of each fit,
            you must specify different `path` locations or don't specify `path` at all.
            Otherwise files from first `fit()` will be overwritten by second `fit()`.
        verbosity
            Verbosity levels range from 0 to 4 and control how much information is printed.
            Higher levels correspond to more detailed print statements (you can set verbosity = 0 to suppress warnings).
            If using logging, you can alternatively control amount of information printed via `logger.setLevel(L)`,
            where `L` ranges from 0 to 50
            (Note: higher values of `L` correspond to fewer print statements, opposite of verbosity levels)
        warn_if_exist
            Whether to raise warning if the specified path already exists.
        enable_progress_bar
            Whether to show progress bar. It will be True by default and will also be
            disabled if the environment variable os.environ["AUTOMM_DISABLE_PROGRESS_BAR"] is set.
        ensemble_size
            A multiple of number of models in the ensembling pool (Default 2). The actual ensemble size = ensemble_size * the model number
        ensemble_mode
            The mode of conducting ensembling:
            - `one_shot`: the classic ensemble selection
            - `sequential`: iteratively calling the classic ensemble selection with each time growing the model zoo by the best next model.
        """
        super().__init__(
            label=label,
            problem_type=problem_type,
            presets=presets,
            eval_metric=eval_metric,
            hyperparameters=hyperparameters,
            path=path,
            verbosity=verbosity,
            warn_if_exist=warn_if_exist,
            enable_progress_bar=enable_progress_bar,
        )
        self._ensemble_size = int(ensemble_size)
        assert ensemble_mode in ["sequential", "one_shot"]
        self._ensemble_mode = ensemble_mode
        self._weighted_ensemble = None
        self._selected_learners = None
        self._all_learners = None
        self._selected_indices = []
        self._relative_path = False

        return

    def get_learner_path(self, learner_path: str):
        if self._relative_path:
            learner_path = os.path.join(self._save_path, learner_path)
        return learner_path

    def get_learner_name(self, learner):
        if isinstance(learner, str):
            if self._relative_path:
                learner_name = learner
            else:
                learner_name = pathlib.PurePath(learner).name
        else:
            learner_name = pathlib.PurePath(learner.path).name

        return learner_name

    def predict_all_for_ensembling(
        self,
        learners: List[Union[str, BaseLearner]],
        data: Union[pd.DataFrame, str],
        mode: str,
        requires_label: Optional[bool] = False,
        save: Optional[bool] = False,
    ):
        assert mode in [VAL, TEST]
        predictions = []
        labels = None
        i = 0
        for per_learner in learners:
            i += 1
            logger.info(f"\npredicting with learner {i}: {per_learner}\n")
            if isinstance(per_learner, str):
                per_learner_path = self.get_learner_path(per_learner)
            else:
                per_learner_path = per_learner.path

            pred_file_path = os.path.join(per_learner_path, f"{mode}_predictions.npy")
            if os.path.isfile(pred_file_path):
                logger.info(f"{mode}_predictions.npy exists. loading it...")
                y_pred = np.load(pred_file_path)
            else:
                if isinstance(per_learner, str):
                    per_learner = BaseLearner.load(path=per_learner_path)
                if not self._problem_type:
                    self._problem_type = per_learner.problem_type
                else:
                    assert self._problem_type == per_learner.problem_type
                outputs = per_learner.predict_per_run(
                    data=data,
                    realtime=False,
                    requires_label=False,
                )
                y_pred = extract_from_output(outputs=outputs, ret_type=LOGITS)

                if self._problem_type == REGRESSION:
                    y_pred = per_learner._df_preprocessor.transform_prediction(y_pred=y_pred)
                if self._problem_type in [BINARY, MULTICLASS]:
                    y_pred = logits_to_prob(y_pred)
                    if self._problem_type == BINARY:
                        y_pred = y_pred[:, 1]

                if save:
                    np.save(pred_file_path, y_pred)

            if requires_label:
                label_file_path = os.path.join(per_learner_path, f"{mode}_labels.npy")
                if os.path.isfile(label_file_path):
                    logger.info(f"{mode}_labels.npy exists. loading it...")
                    y_true = np.load(label_file_path)
                else:
                    if isinstance(per_learner, str):
                        per_learner = BaseLearner.load(path=per_learner_path)
                    y_true = per_learner._df_preprocessor.transform_label_for_metric(df=data)

                    if save:
                        np.save(label_file_path, y_true)

                if labels is None:
                    labels = y_true
                else:
                    assert np.array_equal(y_true, labels)

            predictions.append(y_pred)

        if requires_label:
            return predictions, labels
        else:
            return predictions

    @staticmethod
    def verify_predictions_labels(predictions, learners, labels=None):
        if labels is not None:
            assert isinstance(labels, np.ndarray)
        assert isinstance(predictions, list) and all(isinstance(ele, np.ndarray) for ele in predictions)
        assert len(learners) == len(
            predictions
        ), f"len(learners) {len(learners)} doesn't match len(predictions) {len(predictions)}"

    def fit_per_ensemble(
        self,
        predictions: List[np.ndarray],
        labels: np.ndarray,
    ):
        weighted_ensemble = EnsembleSelection(
            ensemble_size=self._ensemble_size * len(predictions),
            problem_type=self._problem_type,
            metric=self._eval_metric_func,
        )
        weighted_ensemble.fit(predictions=predictions, labels=labels)

        return weighted_ensemble

    def select_next_best(self, left_learner_indices, selected_learner_indices, predictions, labels):
        best_regret = None
        best_weighted_ensemble = None
        best_next_index = None
        for i in left_learner_indices:
            tmp_learner_indices = selected_learner_indices + [i]
            tmp_predictions = [predictions[j] for j in tmp_learner_indices]
            tmp_weighted_ensemble = self.fit_per_ensemble(
                predictions=tmp_predictions,
                labels=labels,
            )
            if best_regret is None or tmp_weighted_ensemble.train_score_ < best_regret:
                best_regret = tmp_weighted_ensemble.train_score_
                best_weighted_ensemble = tmp_weighted_ensemble
                best_next_index = i

        return best_regret, best_next_index, best_weighted_ensemble

    def sequential_ensemble(
        self,
        predictions: List[np.ndarray],
        labels: np.ndarray,
    ):
        selected_learner_indices = []
        all_learner_indices = list(range(len(predictions)))
        best_regret = None
        best_weighted_ensemble = None
        best_selected_learner_indices = None
        while len(selected_learner_indices) < len(all_learner_indices):
            left_learner_indices = [i for i in all_learner_indices if i not in selected_learner_indices]
            assert sorted(all_learner_indices) == sorted(selected_learner_indices + left_learner_indices)
            logger.debug(f"\nleft_learner_indices: {left_learner_indices}")
            if not left_learner_indices:
                break
            logger.debug(f"selected_learner_indices: {selected_learner_indices}")
            tmp_reget, next_index, tmp_weighted_ensemble = self.select_next_best(
                left_learner_indices=left_learner_indices,
                selected_learner_indices=selected_learner_indices,
                predictions=predictions,
                labels=labels,
            )
            selected_learner_indices.append(next_index)
            if best_regret is None or tmp_reget < best_regret:
                best_regret = tmp_reget
                best_weighted_ensemble = tmp_weighted_ensemble
                best_selected_learner_indices = copy.deepcopy(selected_learner_indices)
        logger.debug(f"\nbest score: {self._eval_metric_func._optimum-best_regret}")
        logger.debug(f"best_selected_learner_indices: {best_selected_learner_indices}")
        logger.debug(f"best_ensemble_weights: {best_weighted_ensemble.weights_}")

        return best_weighted_ensemble, best_selected_learner_indices

    def update_hyperparameters(self, hyperparameters: Dict):
        if self._hyperparameters and hyperparameters:
            self._hyperparameters.update(hyperparameters)
        elif hyperparameters:
            self._hyperparameters = hyperparameters

        self._hyperparameters = update_ensemble_hyperparameters(
            presets=self._presets,
            provided_hyperparameters=self._hyperparameters,
        )
        # filter out meta-transformer if no local checkpoint path is provided
        if "early_fusion" in self._hyperparameters:
            if self._hyperparameters["early_fusion"]["model.meta_transformer.checkpoint_path"] == "null":
                self._hyperparameters.pop("early_fusion")
                message = (
                    "`early_fusion` will not be used in ensembling because `early_fusion` relies on MetaTransformer, "
                    "but no local MetaTransformer model checkpoint is provided. To use `early_fusion`, "
                    "download its model checkpoints from https://github.com/invictus717/MetaTransformer to local "
                    "and set the checkpoint path as follows:\n"
                    "```python\n"
                    "hyperparameters = {\n"
                    '    "early_fusion": {\n'
                    '        "model.meta_transformer.checkpoint_path": args.meta_transformer_ckpt_path,\n'
                    "    }\n"
                    "}\n"
                    "```\n"
                    "Note that presets `high_quality` (default) and `medium_quality` need the base model, while preset "
                    "`best_quality` requires the large model. Make sure to download the right MetaTransformer version. "
                    "We recommend using the download links under tag `国内下载源` because the corresponding "
                    "downloaded models are not compressed and can be loaded directly.\n"
                )

                logger.warning(message)

    def fit_all(
        self,
        train_data,
        tuning_data,
        hyperparameters,
        column_types,
        holdout_frac,
        time_limit,
        seed,
        standalone,
        clean_ckpts,
    ):
        self._relative_path = True
        self.update_hyperparameters(hyperparameters=hyperparameters)

        learners = []
        assert (
            len(self._hyperparameters) > 1
        ), f"Ensembling requires training more than 1 learners, but got {len(self._hyperparameters)} sets of hyperparameters."
        logger.info(
            f"Will ensemble {len(self._hyperparameters)} models with the following configs:\n {pprint.pformat(self._hyperparameters)}"
        )
        for per_name, per_hparams in self._hyperparameters.items():
            per_learner_path = os.path.join(self._save_path, per_name)
            if not os.path.isdir(per_learner_path):
                logger.info(f"\nfitting learner {per_name}")
                logger.debug(f"hyperparameters: {per_hparams}")
                per_learner = BaseLearner(
                    label=self._label_column,
                    problem_type=self._problem_type,
                    presets=self._presets,
                    eval_metric=self._eval_metric_func,
                    hyperparameters=per_hparams,
                    path=per_learner_path,
                    verbosity=self._verbosity,
                    warn_if_exist=self._warn_if_exist,
                    enable_progress_bar=self._enable_progress_bar,
                    pretrained=self._pretrained,
                    validation_metric=self._validation_metric_name,
                )
                per_learner.fit(
                    train_data=train_data,
                    tuning_data=tuning_data,
                    time_limit=time_limit,
                    column_types=column_types,
                    holdout_frac=holdout_frac,
                    seed=seed,
                    standalone=standalone,
                    clean_ckpts=clean_ckpts,
                )
            learners.append(per_name)

        return learners

    def on_fit_end(
        self,
        training_start: float,
        **kwargs,
    ):
        self._fit_called = True
        training_end = time.time()
        self._total_train_time = training_end - training_start
        logger.info(on_fit_end_message(self._save_path))

    def update_attributes_by_first_learner(self, learners: List):
        # load df preprocessor from the first learner
        if isinstance(learners[0], str):
            first_learner_path = self.get_learner_path(learners[0])
            dir_path, ckpt_path = get_dir_ckpt_paths(path=first_learner_path)
            assert os.path.isdir(dir_path), f"'{dir_path}' must be an existing directory."
            first_learner = BaseLearner(label="dummy_label")
            first_learner = BaseLearner._load_metadata(learner=first_learner, path=dir_path)
        else:
            first_learner = learners[0]

        self._df_preprocessor = first_learner._df_preprocessor
        self._eval_metric_func = first_learner._eval_metric_func
        self._eval_metric_name = first_learner._eval_metric_name
        self._problem_type = first_learner._problem_type

    def fit_ensemble(
        self,
        predictions: Optional[List[np.ndarray]] = None,
        labels: Optional[np.ndarray] = None,
        learners: Optional[List[Union[str, BaseLearner]]] = None,
        train_data: Optional[Union[pd.DataFrame, str]] = None,
        tuning_data: Optional[Union[pd.DataFrame, str]] = None,
        holdout_frac: Optional[float] = None,
        seed: Optional[int] = 0,
    ):
        if not predictions or labels is None:
            self.prepare_train_tuning_data(
                train_data=train_data,
                tuning_data=tuning_data,
                holdout_frac=holdout_frac,
                seed=seed,
            )
            predictions, labels = self.predict_all_for_ensembling(
                learners=learners,
                data=self._tuning_data,
                mode=VAL,
                requires_label=True,
                save=True,
            )

        self.verify_predictions_labels(
            predictions=predictions,
            labels=labels,
            learners=learners,
        )

        if self._ensemble_mode == "sequential":
            weighted_ensemble, selected_learner_indices = self.sequential_ensemble(
                predictions=predictions,
                labels=labels,
            )
        elif self._ensemble_mode == "one_shot":
            weighted_ensemble = self.fit_per_ensemble(
                predictions=predictions,
                labels=labels,
            )
            selected_learner_indices = list(range(len(learners)))
        else:
            raise ValueError(f"Unsupported ensemble_mode: {self._ensemble_mode}")

        predictions = [predictions[j] for j in selected_learner_indices]
        predictions = weighted_ensemble.predict_proba(predictions)

        # for regression, the transform_prediction() is already called in predict_all()
        if self._eval_metric_func.needs_pred and self._problem_type != REGRESSION:
            predictions = self._df_preprocessor.transform_prediction(
                y_pred=predictions,
                inverse_categorical=False,
            )
        metric_data = {
            Y_PRED: predictions,
            Y_TRUE: labels,
        }
        score = compute_score(
            metric_data=metric_data,
            metric=self._eval_metric_func,
        )

        logger.debug(f"\nEnsembling score on validation data: {score}")

        return weighted_ensemble, selected_learner_indices

    def fit(
        self,
        train_data: Union[pd.DataFrame, str],
        presets: Optional[str] = None,
        tuning_data: Optional[Union[pd.DataFrame, str]] = None,
        time_limit: Optional[int] = None,
        save_path: Optional[str] = None,
        hyperparameters: Optional[Union[str, Dict, List[str]]] = None,
        column_types: Optional[Dict] = None,
        holdout_frac: Optional[float] = None,
        teacher_learner: Union[str, BaseLearner] = None,
        seed: Optional[int] = 0,
        standalone: Optional[bool] = True,
        hyperparameter_tune_kwargs: Optional[Dict] = None,
        clean_ckpts: Optional[bool] = True,
        learners: Optional[List[Union[str, BaseLearner]]] = None,
        predictions: Optional[List[np.ndarray]] = None,
        labels: Optional[np.ndarray] = None,
        **kwargs,
    ):
        self.setup_save_path(save_path=save_path)
        training_start = self.on_fit_start(presets=presets)
        if learners is None:
            learners = self.fit_all(
                train_data=train_data,
                tuning_data=tuning_data,
                hyperparameters=hyperparameters,
                column_types=column_types,
                holdout_frac=holdout_frac,
                time_limit=time_limit,
                seed=seed,
                standalone=standalone,
                clean_ckpts=clean_ckpts,
            )
        assert len(learners) > 1, f"Ensembling requires more than 1 learners, but got {len(learners)}."

        self.update_attributes_by_first_learner(learners=learners)
        weighted_ensemble, selected_learner_indices = self.fit_ensemble(
            predictions=predictions,
            labels=labels,
            learners=learners,
            train_data=train_data,
            tuning_data=tuning_data,
            holdout_frac=holdout_frac,
            seed=seed,
        )

        assert len(selected_learner_indices) == len(weighted_ensemble.weights_)
        self._weighted_ensemble = weighted_ensemble
        self._selected_learners = [learners[i] for i in selected_learner_indices]
        self._all_learners = learners
        self._selected_indices = selected_learner_indices

        self.on_fit_end(training_start=training_start)
        self.save(path=self._save_path)

        return self

    def predict(
        self,
        data: Union[pd.DataFrame, dict, list, str],
        predictions: Optional[List[np.ndarray]] = None,
        as_pandas: Optional[bool] = None,
        **kwargs,
    ):
        self.on_predict_start()
        if not predictions:
            predictions = self.predict_all_for_ensembling(
                learners=self._selected_learners,
                data=data,
                mode=TEST,
                requires_label=False,
                save=False,
            )
        else:
            predictions = [predictions[i] for i in self._selected_indices]

        self.verify_predictions_labels(
            predictions=predictions,
            learners=self._selected_learners,
        )
        pred = self._weighted_ensemble.predict_proba(predictions)
        # for regression, the transform_prediction() is already called in predict_all()
        if self._problem_type in [BINARY, MULTICLASS]:
            pred = self._df_preprocessor.transform_prediction(
                y_pred=pred,
                inverse_categorical=True,
            )
        if (as_pandas is None and isinstance(data, pd.DataFrame)) or as_pandas is True:
            pred = self._as_pandas(data=data, to_be_converted=pred)

        return pred

    def predict_proba(
        self,
        data: Union[pd.DataFrame, dict, list],
        predictions: Optional[List[np.ndarray]] = None,
        as_pandas: Optional[bool] = None,
        as_multiclass: Optional[bool] = True,
        **kwargs,
    ):
        self.on_predict_start()
        assert self._problem_type not in [
            REGRESSION,
        ], f"Problem {self._problem_type} has no probability output."

        if not predictions:
            predictions = self.predict_all_for_ensembling(
                learners=self._selected_learners,
                data=data,
                mode=TEST,
                requires_label=False,
                save=False,
            )
        else:
            predictions = [predictions[i] for i in self._selected_indices]

        self.verify_predictions_labels(
            predictions=predictions,
            learners=self._selected_learners,
        )
        prob = self._weighted_ensemble.predict_proba(predictions)
        if as_multiclass and self._problem_type == BINARY:
            prob = np.column_stack((1 - prob, prob))

        if (as_pandas is None and isinstance(data, pd.DataFrame)) or as_pandas is True:
            prob = self._as_pandas(data=data, to_be_converted=prob)

        return prob

    def evaluate(
        self,
        data: Union[pd.DataFrame, dict, list, str],
        predictions: Optional[List[np.ndarray]] = None,
        labels: Optional[np.ndarray] = None,
        save_all: Optional[bool] = True,
        **kwargs,
    ):
        self.on_predict_start()
        if not predictions or labels is None:
            if save_all:
                learners = self._all_learners
            else:
                learners = self._selected_learners
            predictions, labels = self.predict_all_for_ensembling(
                learners=learners,
                data=data,
                mode=TEST,
                requires_label=True,
                save=True,
            )
            if save_all:
                predictions = [predictions[i] for i in self._selected_indices]
        else:
            predictions = [predictions[i] for i in self._selected_indices]

        self.verify_predictions_labels(
            predictions=predictions,
            labels=labels,
            learners=self._selected_learners,
        )
        all_scores = dict()
        for per_predictions, per_learner in zip(predictions, self._selected_learners):
            if not isinstance(per_learner, str):
                per_learner = per_learner.path
            metric_data = {
                Y_PRED: per_predictions,
                Y_TRUE: labels,
            }
            all_scores[per_learner] = compute_score(
                metric_data=metric_data,
                metric=self._eval_metric_func,
            )

        predictions = self._weighted_ensemble.predict_proba(predictions)
        # for regression, the transform_prediction() is already called in predict_all()
        if self._eval_metric_func.needs_pred and self._problem_type != REGRESSION:
            predictions = self._df_preprocessor.transform_prediction(
                y_pred=predictions,
                inverse_categorical=False,
            )
        metric_data = {
            Y_PRED: predictions,
            Y_TRUE: labels,
        }
        all_scores["ensemble"] = compute_score(
            metric_data=metric_data,
            metric=self._eval_metric_func,
        )

        return all_scores

    def extract_embedding(
        self,
        data: Union[pd.DataFrame, dict, list],
        return_masks: Optional[bool] = False,
        as_tensor: Optional[bool] = False,
        as_pandas: Optional[bool] = False,
        realtime: Optional[bool] = False,
        **kwargs,
    ):
        raise ValueError(f"EnsembleLearner doesn't support extracting embedding yet.")

    def save(
        self,
        path: str,
        **kwargs,
    ):
        selected_learner_names = [self.get_learner_name(per_learner) for per_learner in self._selected_learners]
        all_learner_names = [self.get_learner_name(per_learner) for per_learner in self._all_learners]

        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, f"assets.json"), "w") as fp:
            json.dump(
                {
                    "learner_class": self.__class__.__name__,
                    "ensemble_size": self._ensemble_size,
                    "ensemble_mode": self._ensemble_mode,
                    "selected_learners": selected_learner_names,
                    "all_learners": all_learner_names,
                    "selected_indices": self._selected_indices,
                    "ensemble_weights": self._weighted_ensemble.weights_.tolist(),
                    "save_path": path,
                    "relative_path": True,
                    "fit_called": self._fit_called,
                    "version": ag_version.__version__,
                    "hyperparameters": self._hyperparameters,
                },
                fp,
                ensure_ascii=True,
            )

        with open(os.path.join(path, "ensemble.pkl"), "wb") as fp:
            pickle.dump(self._weighted_ensemble, fp)

        # save each learner
        for per_learner in self._all_learners:
            per_learner_name = self.get_learner_name(per_learner)
            if isinstance(per_learner, str):
                per_learner_path = self.get_learner_path(per_learner)
                per_learner = BaseLearner.load(per_learner_path)

            per_learner_save_path = os.path.join(path, per_learner_name)
            per_learner.save(per_learner_save_path)

        return

    @classmethod
    def load(
        cls,
        path: str,
        **kwargs,
    ):
        dir_path, ckpt_path = get_dir_ckpt_paths(path=path)
        assert os.path.isdir(dir_path), f"'{dir_path}' must be an existing directory."
        with open(os.path.join(dir_path, "assets.json"), "r") as fp:
            assets = json.load(fp)

        learner = cls(
            hyperparameters=assets["hyperparameters"],
        )
        learner._ensemble_size = assets["ensemble_size"]
        learner._ensemble_mode = assets["ensemble_mode"]
        learner._selected_learners = assets["selected_learners"]
        learner._all_learners = assets["all_learners"]
        learner._selected_indices = assets["selected_indices"]
        learner._save_path = path  # in case the original exp dir is copied to somewhere else
        learner._relative_path = assets["relative_path"]
        learner._fit_called = assets["fit_called"]

        with open(os.path.join(path, "ensemble.pkl"), "rb") as fp:
            learner._weighted_ensemble = pickle.load(fp)  # nosec B301

        learner.update_attributes_by_first_learner(learners=learner._selected_learners)

        return learner
