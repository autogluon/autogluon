import logging
from typing import Callable, Dict, List, Optional, Union

from omegaconf import DictConfig, OmegaConf
from torch import nn

from ..constants import REGRESSION
from ..data import turn_on_off_feature_column_info
from ..models import modify_duplicate_model_names
from ..optim.losses import RKDLoss

logger = logging.getLogger(__name__)


class DistillationMixin:
    def setup_distillation(
        self,
        model: nn.Module,
        loss_func: Callable,
        config: DictConfig,
        data_processors: Dict,
    ):
        """
        Prepare for distillation. It verifies whether the student and teacher learners have consistent
        configurations. If teacher and student have duplicate model names, it modifies teacher's model names.

        Returns
        -------
        distillation_kwargs
            Distillation related keyword arguments.
        """
        if self._teacher_learner is None:
            return dict()
        logger.debug("setting up distillation...")
        if isinstance(self._teacher_learner, str):
            from ..learners.base import BaseLearner

            self._teacher_learner = BaseLearner.load(self._teacher_learner)

        # verify that student and teacher configs are consistent.
        assert self._problem_type == self._teacher_learner.problem_type
        assert self._label_column == self._teacher_learner._label_column
        assert self._output_shape == self._teacher_learner._output_shape

        # if teacher and student have duplicate model names, change teacher's model names
        # we don't change student's model names to avoid changing the names back when saving the model.
        self._teacher_learner = modify_duplicate_model_names(
            learner=self._teacher_learner,
            postfix="teacher",
            blacklist=config.model.names,
        )

        critics, baseline_funcs = None, None
        if not config.distiller.soft_label_loss_type:
            # automatically infer loss func based on problem type if not specified
            if self._problem_type == REGRESSION:
                soft_label_loss_func = nn.MSELoss()
            else:
                assert self._output_shape > 1
                soft_label_loss_func = nn.CrossEntropyLoss()
        elif config.distiller.soft_label_loss_type == "mse":
            soft_label_loss_func = nn.MSELoss()
        elif config.distiller.soft_label_loss_type == "cross_entropy":
            soft_label_loss_func = nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Unknown soft_label_loss_type: {config.distiller.soft_label_loss_type}")

        if not config.distiller.softmax_regression_loss_type:
            # automatically infer loss func based on problem type if not specified
            if self._problem_type == REGRESSION:
                softmax_regression_loss_func = nn.MSELoss()
            else:
                assert self._output_shape > 1
                softmax_regression_loss_func = nn.CrossEntropyLoss()
        elif config.distiller.softmax_regression_loss_type == "mse":
            softmax_regression_loss_func = nn.MSELoss()
        elif config.distiller.softmax_regression_loss_type == "cross_entropy":
            softmax_regression_loss_func = nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Unknown soft_label_loss_type: {config.distiller.softmax_regression_loss_type}")

        output_feature_loss_type = config.distiller.output_feature_loss_type
        if output_feature_loss_type == "cosine":
            output_feature_loss_func = nn.CosineEmbeddingLoss()
        elif output_feature_loss_type == "mse":
            output_feature_loss_func = nn.MSELoss()
        else:
            raise ValueError(f"Unknown output_feature_loss_type: {output_feature_loss_type}")

        # Adapt student's output_feature feature to teacher's
        # Refer to FitNet: https://arxiv.org/abs/1412.6550
        teacher_model_dim = self._teacher_learner._model.out_features
        student_model_dim = model.out_features
        output_feature_adaptor = (
            nn.Linear(student_model_dim, teacher_model_dim)
            if teacher_model_dim != student_model_dim
            else nn.Identity()
        )

        rkd_distance_loss_weight = config.distiller.rkd_distance_loss_weight
        rkd_angle_loss_weight = config.distiller.rkd_angle_loss_weight
        rkd_loss_func = RKDLoss(rkd_distance_loss_weight, rkd_angle_loss_weight)
        output_feature_loss_weight = config.distiller.output_feature_loss_weight
        softmax_regression_weight = config.distiller.softmax_regression_weight

        # turn on returning column information in data processors
        turn_on_off_feature_column_info(
            data_processors=data_processors,
            flag=True,
        )
        turn_on_off_feature_column_info(
            data_processors=self._teacher_learner._data_processors,
            flag=True,
        )

        return dict(
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
        )
