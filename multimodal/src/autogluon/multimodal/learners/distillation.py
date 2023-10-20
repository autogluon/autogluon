from typing import List, Optional, Union

import pandas as pd
import pytorch_lightning as pl
from omegaconf import OmegaConf
from timm.data.mixup import Mixup
from torch import nn

from autogluon.core.utils.loaders import load_pd

from ..constants import NER
from ..optimization.lit_ner import NerLitModule
from ..utils.model import create_fusion_model, select_model
from ..utils.object_detection import setup_detection_train_tuning_data
from .base_learner import BaseLearner


class DistillationLearner(BaseLearner):
    def __init__(
        self,
        label: Optional[str] = None,
        problem_type: Optional[str] = NER,
        presets: Optional[str] = None,
        eval_metric: Optional[str] = None,
        hyperparameters: Optional[dict] = None,
        path: Optional[str] = None,
        verbosity: Optional[int] = 2,
        num_classes: Optional[int] = None,  # TODO: can we infer this from data?
        warn_if_exist: Optional[bool] = True,
        enable_progress_bar: Optional[bool] = None,
        pretrained: Optional[bool] = True,
    ):
        assert problem_type == NER, f"Expected problem_type={NER}, but problem_type={problem_type}"
        super().__init__(
            label=label,
            problem_type=problem_type,
            presets=presets,
            eval_metric=eval_metric,
            hyperparameters=hyperparameters,
            path=path,
            verbosity=verbosity,
            num_classes=num_classes,
            warn_if_exist=warn_if_exist,
            enable_progress_bar=enable_progress_bar,
            pretrained=pretrained,
        )

    def _get_lightning_module(
        self,
        trainable_param_names: List[str],
        mixup_fn: Optional[Mixup] = None,
        loss_func: Optional[nn.Module] = None,
        optimization_kwargs: Optional[dict] = None,
        metrics_kwargs: Optional[dict] = None,
        test_time: bool = False,
        **kwargs,
    ):
        if test_time:
            task = NerLitModule(
                model=self._model,
                model_postprocess_fn=self._model_postprocess_fn,
                efficient_finetune=OmegaConf.select(self._config, "optimization.efficient_finetune"),
                trainable_param_names=trainable_param_names,
                **optimization_kwargs,
            )
        else:
            task = NerLitModule(
                model=self._model,
                loss_func=loss_func,
                efficient_finetune=OmegaConf.select(self._config, "optimization.efficient_finetune"),
                mixup_fn=mixup_fn,
                mixup_off_epoch=OmegaConf.select(self._config, "data.mixup.turn_off_epoch"),
                model_postprocess_fn=self._model_postprocess_fn,
                trainable_param_names=trainable_param_names,
                **metrics_kwargs,
                **optimization_kwargs,
            )
        return task

    def _setup_distillation(
            self,
            teacher_learner: Union[str, BaseLearner],
    ):
        """
        Prepare for distillation. It verifies whether the student and teacher predictors have consistent
        configurations. If teacher and student have duplicate model names, it modifies teacher's model names.

        Parameters
        ----------
        teacher_predictor
            The teacher predictor in knowledge distillation.

        Returns
        -------
        teacher_model
            The teacher predictor's model.
        critics
            The critics used in computing mutual information loss.
        baseline_funcs
            The baseline functions used in computing mutual information loss.
        soft_label_loss_func
            The loss function using teacher's logits as labels.
        output_feature_adaptor
            The adaptor used to adapt student output feature to the shape of teacher's.
        output_feature_loss_func
            The loss function using minimize distance between output_feature of teacher and student.
        rkd_loss_func
            The loss function using rkd distance and angle loss between output_feature of teacher and student.
        df_preprocessor
            The teacher predictor's dataframe preprocessor.
        data_processors
            The teacher predictor's data processors.
        """
        logger.debug("setting up distillation...")
        if isinstance(teacher_learner, str):
            teacher_learner = BaseLearner.load(teacher_learner)

        # verify that student and teacher configs are consistent.
        assert self._problem_type == teacher_predictor.problem_type
        assert self._label_column == teacher_predictor._label_column
        assert self._output_shape == teacher_predictor._output_shape

        # if teacher and student have duplicate model names, change teacher's model names
        # we don't change student's model names to avoid changing the names back when saving the model.
        teacher_predictor = modify_duplicate_model_names(
            predictor=teacher_predictor,
            postfix="teacher",
            blacklist=self._config.model.names,
        )

        critics, baseline_funcs = None, None
        if not self._config.distiller.soft_label_loss_type:
            # automatically infer loss func based on problem type if not specified
            if self._problem_type == REGRESSION:
                soft_label_loss_func = nn.MSELoss()
            else:
                assert self._output_shape > 1
                soft_label_loss_func = nn.CrossEntropyLoss()
        elif self._config.distiller.soft_label_loss_type == "mse":
            soft_label_loss_func = nn.MSELoss()
        elif self._config.distiller.soft_label_loss_type == "cross_entropy":
            soft_label_loss_func = nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Unknown soft_label_loss_type: {self._config.distiller.soft_label_loss_type}")

        if not self._config.distiller.softmax_regression_loss_type:
            # automatically infer loss func based on problem type if not specified
            if self._problem_type == REGRESSION:
                softmax_regression_loss_func = nn.MSELoss()
            else:
                assert self._output_shape > 1
                softmax_regression_loss_func = nn.CrossEntropyLoss()
        elif self._config.distiller.softmax_regression_loss_type == "mse":
            softmax_regression_loss_func = nn.MSELoss()
        elif self._config.distiller.softmax_regression_loss_type == "cross_entropy":
            softmax_regression_loss_func = nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Unknown soft_label_loss_type: {self._config.distiller.softmax_regression_loss_type}")

        output_feature_loss_type = OmegaConf.select(self._config, "distiller.output_feature_loss_type", default="mse")
        if output_feature_loss_type == "cosine":
            output_feature_loss_func = nn.CosineEmbeddingLoss()
        elif output_feature_loss_type == "mse":
            output_feature_loss_func = nn.MSELoss()
        else:
            raise ValueError(f"Unknown output_feature_loss_type: {output_feature_loss_type}")

        # Adapt student's output_feature feature to teacher's
        # Refer to FitNet: https://arxiv.org/abs/1412.6550
        teacher_model_dim = teacher_predictor._model.out_features
        student_model_dim = self._model.out_features
        output_feature_adaptor = (
            nn.Linear(student_model_dim, teacher_model_dim)
            if teacher_model_dim != student_model_dim
            else nn.Identity()
        )

        rkd_distance_loss_weight = OmegaConf.select(self._config, "distiller.rkd_distance_loss_weight", default=0.0)
        rkd_angle_loss_weight = OmegaConf.select(self._config, "distiller.rkd_angle_loss_weight", default=0.0)
        rkd_loss_func = RKDLoss(rkd_distance_loss_weight, rkd_angle_loss_weight)

        # turn on returning column information in data processors
        turn_on_off_feature_column_info(
            data_processors=self._data_processors,
            flag=True,
        )
        turn_on_off_feature_column_info(
            data_processors=teacher_predictor._data_processors,
            flag=True,
        )

        return (
            teacher_predictor._model,
            critics,
            baseline_funcs,
            soft_label_loss_func,
            softmax_regression_loss_func,
            output_feature_adaptor,
            output_feature_loss_func,
            rkd_loss_func,
            teacher_predictor._df_preprocessor,
            teacher_predictor._data_processors,
        )

    def build_task_per_run(self, model, teacher_model, critics, baseline_funcs, loss_func, config, mixup_func, model_postprocess_fn, peft_param_names, optimization_kwargs):
        output_feature_loss_weight = OmegaConf.select(
            self._config, "distiller.output_feature_loss_weight", default=0.0
        )
        softmax_regression_weight = OmegaConf.select(
            self._config, "distiller.softmax_regression_weight", default=0.0
        )
        use_raw_features = OmegaConf.select(self._config, "distiller.use_raw_features", default=False)
        task = DistillerLitModule(
            student_model=model,
            teacher_model=teacher_model,
            matches=config.distiller.matches,
            critics=critics,
            baseline_funcs=baseline_funcs,
            hard_label_weight=config.distiller.hard_label_weight,
            soft_label_weight=config.distiller.soft_label_weight,
            softmax_regression_weight=softmax_regression_weight,
            temperature=config.distiller.temperature,
            output_feature_loss_weight=output_feature_loss_weight,
            hard_label_loss_func=loss_func,
            soft_label_loss_func=soft_label_loss_func,
            softmax_regression_loss_func=softmax_regression_loss_func,
            output_feature_adaptor=output_feature_adaptor,
            output_feature_loss_func=output_feature_loss_func,
            rkd_loss_func=rkd_loss_func,
            **optimization_kwargs,
        )
        return task

    def fit_per_run(self):
        (
            teacher_model,
            critics,
            baseline_funcs,
            soft_label_loss_func,
            softmax_regression_loss_func,
            output_feature_adaptor,
            output_feature_loss_func,
            rkd_loss_func,
            teacher_df_preprocessor,
            teacher_data_processors,
        ) = self._setup_distillation(
            teacher_predictor=teacher_learner,
        )
        df_preprocessor = [df_preprocessor, teacher_df_preprocessor]
        data_processors = [data_processors, teacher_data_processors]

