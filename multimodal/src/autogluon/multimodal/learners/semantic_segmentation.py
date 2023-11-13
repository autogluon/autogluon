import logging
import os
from typing import Dict, Iterable, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchmetrics
from omegaconf import OmegaConf
from PIL import Image
from scipy.special import softmax

from autogluon.core.metrics import Scorer

from ..constants import LABEL, LOGITS, MASK_SEMANTIC_INFER, SEMANTIC_SEGMENTATION
from ..models import get_model_postprocess_fn
from ..optimization.lit_semantic_segmentation import SemanticSegmentationLitModule
from ..optimization.utils import (
    get_loss_func,
    get_metric,
    get_norm_layer_param_names,
    get_trainable_params_efficient_finetune,
)
from ..utils import (
    create_fusion_data_processors,
    create_fusion_model,
    extract_from_output,
    get_dir_ckpt_paths,
    get_load_ckpt_paths,
    setup_save_path,
)
from .base import BaseLearner

logger = logging.getLogger(__name__)


class SemanticSegmentationLearner(BaseLearner):
    def __init__(
        self,
        label: Optional[str] = None,
        problem_type: Optional[str] = SEMANTIC_SEGMENTATION,
        presets: Optional[str] = None,
        eval_metric: Optional[Union[str, Scorer]] = None,
        hyperparameters: Optional[dict] = None,
        path: Optional[str] = None,
        verbosity: Optional[int] = 2,
        num_classes: Optional[int] = None,  # TODO: can we infer this from data?
        classes: Optional[list] = None,
        warn_if_exist: Optional[bool] = True,
        enable_progress_bar: Optional[bool] = None,
        pretrained: Optional[bool] = True,
        validation_metric: Optional[str] = None,
        sample_data_path: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            label=label,
            problem_type=problem_type,
            presets=presets,
            eval_metric=eval_metric,
            hyperparameters=hyperparameters,
            path=path,
            verbosity=verbosity,
            num_classes=num_classes,
            classes=classes,
            warn_if_exist=warn_if_exist,
            enable_progress_bar=enable_progress_bar,
            pretrained=pretrained,
            validation_metric=validation_metric,
            sample_data_path=sample_data_path,
        )
        self._output_shape = num_classes
        self._sample_data_path = sample_data_path

        if self._sample_data_path is not None:
            infer_output_shape = self.get_semantic_segmentation_class_num(self._sample_data_path)
            if num_classes is not None:
                assert (
                    num_classes == infer_output_shape
                ), "The provided number of classes and the inferred class number from the sample data should be consistent."
            else:
                self._output_shape = infer_output_shape

    def get_semantic_segmentation_class_num(self, sample_data_path):
        """
        Get the number of classes for given data.

        Parameters
        ----------
            sample_data_path
                This is used for automatically inference num_classes of semantic segmentation dataset.
                Could be an image directory, image file or pd.DataFrame.
        Returns
        -------
            The number of classes.
        """
        if isinstance(sample_data_path, str):
            if os.path.isdir(sample_data_path):
                mask_files = os.listdir(sample_data_path)
                num_classes = []
                for mask_file in mask_files:
                    per_num_classes = self.get_semantic_segmentation_class_num(
                        os.path.join(sample_data_path, mask_file)
                    )
                    num_classes.append(per_num_classes)
                return max(num_classes)
            else:
                mask = Image.open(sample_data_path)
                mode = mask.mode
                if mode == "L":
                    return 1
                elif mode == "P":
                    classes = np.unique(mask)
                    return max(classes).item() + 1  # include background
                else:
                    NotImplementedError(
                        f"Current image mode '{mode}' is not supported. 'P' (Palette) mode and 'L' (Luminance) mode are supported."
                    )

        elif isinstance(sample_data_path, pd.DataFrame):
            num_classes = []
            for idx in range(sample_data_path.shape[0]):
                row = sample_data_path.iloc[idx]
                mask_file = row["label"]
                per_num_classes = self.get_semantic_segmentation_class_num(mask_file)
                num_classes.append(per_num_classes)
            return max(num_classes)

    def infer_output_shape(self, train_data: pd.DataFrame):
        if self._output_shape is None:
            self._output_shape = self.get_semantic_segmentation_class_num(train_data)

    @staticmethod
    def get_peft_param_names_per_run(model, config):
        peft_param_names = None
        peft = OmegaConf.select(config, "optimization.efficient_finetune")
        if peft:
            norm_param_names = get_norm_layer_param_names(model)
            peft_param_names = get_trainable_params_efficient_finetune(
                norm_param_names,
                efficient_finetune=peft,
                extra_params=OmegaConf.select(config, "optimization.extra_trainable_params"),
            )
        return peft_param_names

    def get_loss_func_per_run(self, config, mixup_active=None):
        loss_func = get_loss_func(
            problem_type=self._problem_type,
            mixup_active=mixup_active,
            loss_func_name=OmegaConf.select(config, "optimization.loss_function"),
            config=config.optimization,
            num_classes=self._output_shape,
        )
        return loss_func

    def evaluate_semantic_segmentation(
        self,
        data: Union[pd.DataFrame, dict, list, str],
        metrics: Optional[Union[str, List[str]]] = None,
        return_pred: Optional[bool] = False,
        realtime: Optional[bool] = None,
    ):
        """
        Evaluate semantic segmentation on a test dataset based on "torchmetrics".

        Parameters
        ----------
        data
            A dataframe, containing the same columns as the training data.
            Or a str, that is a path of the annotation file for detection.
        metrics
            Metrics used for evaluation.
        return_pred
            Whether to return the prediction result of each row.
        realtime
            Whether to do realtime inference, which is efficient for small data (default None).
            If not specified, we would infer it on based on the data modalities
            and sample number.
        """
        outputs = self.predict_per_run(
            data=data,
            requires_label=True,
            realtime=realtime,
        )

        if self._output_shape == 1:
            logits = extract_from_output(ret_type=LOGITS, outputs=outputs, as_ndarray=False)
        else:
            logits = extract_from_output(ret_type=MASK_SEMANTIC_INFER, outputs=outputs, as_ndarray=False)
        y_pred = logits.float()
        y_true = [ele[LABEL] for ele in outputs]
        y_true = torch.cat(y_true)

        assert len(y_true) == len(y_pred)

        results = {}
        for per_metric_name in metrics:
            per_metric, _ = get_metric(metric_name=per_metric_name.lower(), num_classes=self._output_shape)
            if isinstance(per_metric, torchmetrics.classification.MulticlassJaccardIndex):
                bs, num_classes = y_pred.shape[0:2]
                y_pred = y_pred.reshape(bs, num_classes, -1)
                y_true = y_true.reshape(bs, -1)
            per_metric.update(y_pred, y_true)
            score = per_metric.compute()

            results[per_metric_name] = score.item()

        if return_pred:
            return results, outputs
        else:
            return results

    def get_litmodule_per_run(
        self,
        model=None,
        model_postprocess_fn=None,
        peft_param_names=None,
        optimization_kwargs=None,
        distillation_kwargs=None,
        is_train=True,
    ):
        if is_train:
            return SemanticSegmentationLitModule(
                model=model,
                model_postprocess_fn=model_postprocess_fn,
                trainable_param_names=peft_param_names,
                **optimization_kwargs,
            )
        else:
            return SemanticSegmentationLitModule(
                model=self._model,
                model_postprocess_fn=self._model_postprocess_fn,
                **optimization_kwargs,
            )

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
        **kwargs,
    ):
        training_start = self.on_fit_start(presets=presets, teacher_learner=teacher_learner)
        self.setup_save_path(save_path=save_path)
        self.infer_problem_type(train_data=train_data)
        self.prepare_train_tuning_data(
            train_data=train_data,
            tuning_data=tuning_data,
            holdout_frac=holdout_frac,
            seed=seed,
        )
        self.infer_column_types(column_types=column_types)
        self.infer_output_shape(train_data)
        self.infer_validation_metric()
        self.update_hyperparameters(
            hyperparameters=hyperparameters,
            hyperparameter_tune_kwargs=hyperparameter_tune_kwargs,
        )
        self.fit_sanity_check()
        self.prepare_fit_args(
            time_limit=time_limit,
            seed=seed,
            standalone=standalone,
            clean_ckpts=clean_ckpts,
        )
        fit_returns = self.execute_fit()
        self.on_fit_end(
            training_start=training_start,
            strategy=fit_returns.get("strategy", None),
            strict_loading=fit_returns.get("strict_loading", True),
            standalone=standalone,
            clean_ckpts=clean_ckpts,
        )

        return self

    def on_predict_start(self, data: pd.DataFrame):
        if self._output_shape is None:  # for zero-shot evaluation/prediction
            self._output_shape = self.get_semantic_segmentation_class_num(data)
        self.ensure_predict_ready()

    def evaluate(
        self,
        data: Union[pd.DataFrame, dict, list, str],
        metrics: Optional[Union[str, List[str]]] = None,
        return_pred: Optional[bool] = False,
        realtime: Optional[bool] = None,
        **kwargs,
    ):
        """
        Evaluate model on a test dataset.

        Parameters
        ----------
        data
            A dataframe, containing the same columns as the training data.
            Or a str, that is a path of the annotation file for detection.
        metrics
            A list of metric names to report.
            If None, we only return the score for the stored `_eval_metric_name`.
        return_pred
            Whether to return the prediction result of each row.
        realtime
            Whether to do realtime inference, which is efficient for small data (default None).
            If not specified, we would infer it on based on the data modalities
            and sample number.

        Returns
        -------
        A dictionary with the metric names and their corresponding scores.
        Optionally return a dataframe of prediction results.
        """
        self.on_predict_start(data)
        return self.evaluate_semantic_segmentation(data, metrics, realtime)

    def predict(
        self,
        data: Union[pd.DataFrame, dict, list, str],
        realtime: Optional[bool] = None,
        save_results: Optional[bool] = None,
        **kwargs,
    ):
        """
        Predict values for the label column of new data.

        Parameters
        ----------
        data
            The data to make predictions for. Should contain same column names as training data and
            follow same format (except for the `label` column).
        as_pandas
            Whether to return the output as a pandas DataFrame(Series) (True) or numpy array (False).
        realtime
            Whether to do realtime inference, which is efficient for small data (default None).
            If not specified, we would infer it on based on the data modalities
            and sample number.
        save_results
            Whether to save the prediction results (only works for detection now)

        Returns
        -------
        Array of predictions, one corresponding to each row in given dataset.
        """
        self.on_predict_start(data)
        if self._output_shape == 1:
            ret_type = LOGITS
        else:
            ret_type = MASK_SEMANTIC_INFER

        outputs = self.predict_per_run(
            data=data,
            requires_label=False,
            realtime=realtime,
        )
        logits = extract_from_output(outputs=outputs, ret_type=ret_type)

        if self._df_preprocessor:
            pred = self._df_preprocessor.transform_prediction(
                y_pred=logits,
            )

        if ret_type == MASK_SEMANTIC_INFER:
            pred = logits.argmax(axis=1)
        else:
            pred = logits > 0.5

        if save_results:
            self._save_path = setup_save_path(
                old_save_path=self._save_path,
                warn_if_exist=False,
            )
            self.save_segmentation_result(
                pred=pred,
                data=data,
                result_path=self._save_path,
            )

        return pred

        # if (as_pandas is None and isinstance(data, pd.DataFrame)) or as_pandas is True:
        #     # TODO
        #     pred = self._as_pandas(data=data, to_be_converted=pred)

    def predict_proba(
        self,
        data: Union[pd.DataFrame, dict, list],
        as_pandas: Optional[bool] = None,
        as_multiclass: Optional[bool] = True,
        realtime: Optional[bool] = None,
        **kwargs,
    ):
        """
        Predict probabilities class probabilities rather than class labels.
        This is only for the classification. Calling it for regression will throw an exception.

        Parameters
        ----------
        data
            The data to make predictions for. Should contain same column names as training data and
              follow same format (except for the `label` column).
        candidate_data
            The candidate data from which to search the query data's matches.
        as_pandas
            Whether to return the output as a pandas DataFrame(Series) (True) or numpy array (False).
        as_multiclass
            Whether to return the probability of all labels or
            just return the probability of the positive class for binary classification problems.
        realtime
            Whether to do realtime inference, which is efficient for small data (default None).
            If not specified, we would infer it on based on the data modalities
            and sample number.

        Returns
        -------
        Array of predicted class-probabilities, corresponding to each row in the given data.
        When as_multiclass is True, the output will always have shape (#samples, #classes, height, width).
        Otherwise, the output will have shape (#samples, height, width)
        """
        assert (self._output_shape == 1 and as_multiclass == False) or (
            self._output_shape > 1 and as_multiclass == True
        )
        self.on_predict_start(data)

        outputs = self.predict_per_run(
            data=data,
            requires_label=False,
            realtime=realtime,
        )
        if as_multiclass:
            logits = extract_from_output(outputs=outputs, ret_type=MASK_SEMANTIC_INFER)
            prob = softmax(logits, axis=1)
        else:
            logits = extract_from_output(outputs=outputs, ret_type=LOGITS)
            prob = logits

        return prob

    def extract_embedding(
        self,
        data: Union[pd.DataFrame, dict, list],
        as_tensor: Optional[bool] = False,
        as_pandas: Optional[bool] = False,
        realtime: Optional[bool] = None,
        **kwargs,
    ):
        raise NotImplementedError("Semantic segmentation doesn't support calling `extract_embedding` yet.")

    def save_segmentation_result(self, pred: Iterable, data: Union[pd.DataFrame, Dict], result_path: str):
        """
        Saving segmentation results in pd.DataFrame format (per image)

        Parameters
        ----------
        pred
            List containing detection results for one image
        data
            pandas data frame or dict containing the image information to be tested
        result_path
            path to save result
        Returns
        -------
        The paths of the segmentation results as pandas DataFrame
        """

        def show_mask(mask, ax):
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
            h, w = mask.shape[-2:]
            mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
            ax.imshow(mask_image)

        if isinstance(data, dict):
            image_names = data["image"]
        else:
            image_names = data["image"].to_list()
        results = []

        mask_path = os.path.join(result_path, "masks")
        txt_path = os.path.join(result_path, "result.txt")
        os.makedirs(mask_path, exist_ok=True)
        for image_pred, image_name in zip(pred, image_names):
            if self._output_shape == 1:
                mask = Image.fromarray(np.squeeze(image_pred, axis=0))
                per_mask_path = os.path.join(mask_path, os.path.basename(image_name))
                mask.save(per_mask_path)
            else:
                masks = []
                classes = np.unique(image_pred)
                for class_id in classes:
                    if class_id == 0:  # bg
                        continue
                    masks.append(image_pred == class_id)

                for mask in masks:
                    show_mask(mask, plt.gca())
                # mask = Image.fromarray(image_pred, mode="P")  # multi-class
                mask_name = ""
                for i in os.path.basename(image_name).split(".")[:-1]:
                    mask_name += i
                per_mask_path = os.path.join(mask_path, os.path.basename(image_name))
                plt.axis("off")
                plt.savefig(per_mask_path, bbox_inches="tight", dpi=300, pad_inches=0.0)

            results.append([image_name, per_mask_path])

        result_df = pd.DataFrame(results, columns=["image", "mask"])
        result_df.to_csv(txt_path, index=False)
        return result_df

    @classmethod
    def load(
        cls,
        path: str,
        resume: Optional[bool] = False,
        verbosity: Optional[int] = 3,
    ):
        """
        Load a learner object from a directory specified by `path`. The to-be-loaded learner
        can be completely or partially trained by .fit(). If a previous training has completed,
        it will load the checkpoint `model.ckpt`. Otherwise if a previous training accidentally
        collapses in the middle, it can load the `last.ckpt` checkpoint by setting `resume=True`.
        It also supports loading one specific checkpoint given its path.

        Parameters
        ----------
        path
            The directory to load the learner object.
        resume
            Whether to resume training from `last.ckpt`. This is useful when a training was accidentally
            broken during the middle and we want to resume the training from the last saved checkpoint.
        verbosity
            Verbosity levels range from 0 to 4 and control how much information is printed.
            Higher levels correspond to more detailed print statements (you can set verbosity = 0 to suppress warnings).

        Returns
        -------
        The loaded learner object.
        """
        dir_path, ckpt_path = get_dir_ckpt_paths(path=path)

        assert os.path.isdir(dir_path), f"'{dir_path}' must be an existing directory."
        learner = cls(label="dummy_label")
        learner = cls._load_metadata(learner=learner, path=dir_path, resume=resume, verbosity=verbosity)
        peft = OmegaConf.select(learner._config, "optimization.efficient_finetune")
        learner._model = create_fusion_model(
            config=learner._config,
            num_classes=learner._output_shape,
            classes=learner._classes if hasattr(learner, "_classes") else None,
            num_numerical_columns=len(learner._df_preprocessor.numerical_feature_names),
            num_categories=learner._df_preprocessor.categorical_num_categories,
            pretrained=False if not peft else True,  # set "pretrain=False" to prevent downloading online models
        )
        if learner._data_processors is None:
            learner._data_processors = create_fusion_data_processors(
                config=learner._config,
                model=learner._model,
            )
        load_path, ckpt_path = get_load_ckpt_paths(
            ckpt_path=ckpt_path,
            dir_path=dir_path,
            resume=resume,
        )
        learner._load_state_dict(
            path=load_path,
            strict=not peft,
        )
        learner._ckpt_path = ckpt_path
        loss_func = get_loss_func(
            problem_type=learner._problem_type,
            mixup_active=False,
            loss_func_name=OmegaConf.select(learner._config, "optimization.loss_function"),
            config=learner._config.optimization,
            num_classes=learner._output_shape,  # New added. for semantic segmentation
        )
        model_postprocess_fn = get_model_postprocess_fn(
            problem_type=learner._problem_type,
            loss_func=loss_func,
        )
        learner._model_postprocess_fn = model_postprocess_fn
        learner._config = learner.update_strategy_by_env(learner._config)

        return learner
