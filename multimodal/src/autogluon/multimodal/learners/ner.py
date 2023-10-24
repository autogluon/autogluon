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


class NERLearner(BaseLearner):
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

    def build_task_per_run(
        self,
        model,
        config,
        peft_param_names: List[str],
        loss_func: Optional[nn.Module] = None,
        optimization_kwargs: Optional[dict] = None,
        is_train=True,
    ):
        if is_train:
            return NerLitModule(
                model=model,
                loss_func=loss_func,
                efficient_finetune=OmegaConf.select(config, "optimization.efficient_finetune"),
                model_postprocess_fn=self._model_postprocess_fn,
                trainable_param_names=peft_param_names,
                **optimization_kwargs,
            )
        else:
            return NerLitModule(
                model=self._model,
                model_postprocess_fn=self._model_postprocess_fn,
            )

    def get_output_shape_per_run(self, df_preprocessor):
        # ner needs to update output_shape with label_generator.
        return len(df_preprocessor.label_generator.unique_entity_groups)

    def on_fit_per_run_end(self, trainer, model, save_path, config, strategy, peft_param_names, standalone, clean_ckpts):
        if trainer.global_rank == 0:
            # We do not perform averaging checkpoint in the case of hpo for each trial
            # We only average the checkpoint of the best trial at the end in the master process.
            if not self._is_hpo:
                self._top_k_average(
                    model=model,
                    validation_metric_name=OVERALL_F1,  # since we called self.evaluate. Below is a temporal fix for NER. seqeval only support overall_f1
                    save_path=save_path,
                    top_k_average_method=config.optimization.top_k_average_method,
                    strategy=strategy,
                    strict_loading=not peft_param_names,
                    # Not strict loading if using parameter-efficient finetuning
                    standalone=standalone,
                    clean_ckpts=clean_ckpts,
                )
            self._best_score = trainer.callback_metrics[f"val_{self._validation_metric_name}"].item()
        else:
            sys.exit(f"Training finished, exit the process with global_rank={trainer.global_rank}...")

    def fit_per_run(
            self,
            validation_metric_name: str,
            minmax_mode: str,
            max_time: timedelta,
            save_path: str,
            ckpt_path: str,
            resume: bool,
            enable_progress_bar: bool,
            seed: int,
            hyperparameters: Optional[Union[str, Dict, List[str]]] = None,
            advanced_hyperparameters: Optional[Dict] = None,
            config: Optional[Dict] = None,
            df_preprocessor: Optional[MultiModalFeaturePreprocessor] = None,
            data_processors: Optional[Dict] = None,
            model: Optional[nn.Module] = None,
            is_hpo: bool = False,
            standalone: bool = True,
            clean_ckpts: bool = True,
    ):
        pl.seed_everything(seed, workers=True)
        # TODO(?) We should have a separate "_pre_training_event()" for logging messages.
        logger.info(get_fit_start_message(save_path, validation_metric_name))
        config = self.get_config_per_run(config=config, hyperparameters=hyperparameters)
        df_preprocessor = self.get_df_preprocessor_per_run(
            df_preprocessor=df_preprocessor,
            config=config,
        )
        config = self.update_config_by_data_per_run(config=config, df_preprocessor=df_preprocessor)
        output_shape = self.get_output_shape_per_run(df_preprocessor=df_preprocessor)
        model = self.get_model_per_run(model=model, config=config, df_preprocessor=df_preprocessor)

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