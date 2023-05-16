from typing import List, Optional, Union

import pandas as pd
import pytorch_lightning as pl
from omegaconf import OmegaConf
from timm.data.mixup import Mixup
from torch import nn

from autogluon.core.utils.loaders import load_pd

from ..constants import NER
from ..data.infer_types import infer_problem_type_output_shape
from ..optimization.lit_ner import NerLitModule
from ..utils.model import create_fusion_model, select_model
from ..utils.object_detection import setup_detection_train_tuning_data
from .base_learner import BaseLearner


class NERLearner(BaseLearner):
    def __init__(
        self,
        label: Optional[str] = None,
        problem_type: Optional[str] = NER,
        query: Optional[Union[str, List[str]]] = None,
        response: Optional[Union[str, List[str]]] = None,
        match_label: Optional[Union[int, str]] = None,
        pipeline: Optional[str] = None,
        presets: Optional[str] = None,
        eval_metric: Optional[str] = None,
        hyperparameters: Optional[dict] = None,
        path: Optional[str] = None,
        verbosity: Optional[int] = 2,
        num_classes: Optional[int] = None,  # TODO: can we infer this from data?
        classes: Optional[list] = None,
        warn_if_exist: Optional[bool] = True,
        enable_progress_bar: Optional[bool] = None,
        init_scratch: Optional[bool] = False,
        sample_data_path: Optional[str] = None,
    ):
        assert problem_type == NER, f"Expected problem_type={NER}, but problem_type={problem_type}"
        super().__init__(
            label=label,
            problem_type=problem_type,
            query=query,
            response=response,
            match_label=match_label,
            pipeline=pipeline,
            presets=presets,
            eval_metric=eval_metric,
            hyperparameters=hyperparameters,
            path=path,
            verbosity=verbosity,
            num_classes=num_classes,
            classes=classes,
            warn_if_exist=warn_if_exist,
            enable_progress_bar=enable_progress_bar,
            init_scratch=init_scratch,
            sample_data_path=sample_data_path,
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

    def _get_model(self):
        """
        Setup and update config (DictConfig) for the model, and get the model (nn.Module)

        Returns:
            Tuple[DictConfig, nn.Module]: the updated config (DictConfig) and the model (nn.Module)
        """
        config = select_model(config=self._config, df_preprocessor=self._df_preprocessor)

        # 4. if NER, update output shape. TODO: This can be refactored into the NER Learner
        # Update output_shape with label_generator.
        self._output_shape = len(self._df_preprocessor.label_generator.unique_entity_groups)

        # 5. get model
        if self._model is None:
            model = create_fusion_model(
                config=config,
                num_classes=self._output_shape,
                classes=self._classes,
                num_numerical_columns=len(self._df_preprocessor.numerical_feature_names),
                num_categories=self._df_preprocessor.categorical_num_categories,
            )
        else:  # continuing training
            model = self._model

        return config, model
