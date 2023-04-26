from __future__ import annotations

import warnings

from abc import ABC, abstractmethod, abstractclassmethod, abstractstaticmethod, abstractproperty
from datetime import timedelta
from typing import Dict, Optional, Union, List

from .abstract_mm_learner import AbstractMMLearner
from autogluon.multimodal.problem_types import ProblemTypeProperty


import copy
import json
import logging
import operator
import os
import pickle
import shutil
import sys
import time
import warnings
from datetime import timedelta
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import transformers
import yaml
from omegaconf import OmegaConf
from packaging import version
from pytorch_lightning import LightningDataModule
from torch import nn

from autogluon.common.utils.log_utils import set_logger_verbosity, verbosity2loglevel
from autogluon.core.utils import default_holdout_frac, generate_train_test_split_combined
from autogluon.core.utils.loaders import load_pd
from autogluon.multimodal.utils.log import get_fit_complete_message, get_fit_start_message

from . import version as ag_version
from autogluon.multimodal.constants import (
    AUTOMM,
    AUTOMM_TUTORIAL_MODE,
    BBOX,
    BEST,
    BEST_K_MODELS_FILE,
    BINARY,
    COLUMN_FEATURES,
    DEEPSPEED_MIN_PL_VERSION,
    DEEPSPEED_MODULE,
    DEEPSPEED_OFFLOADING,
    DEEPSPEED_STRATEGY,
    DEPRECATED_ZERO_SHOT,
    DOCUMENT,
    FEATURE_EXTRACTION,
    FEATURES,
    FEW_SHOT,
    FEW_SHOT_TEXT_CLASSIFICATION,
    GREEDY_SOUP,
    IMAGE_BYTEARRAY,
    IMAGE_PATH,
    LABEL,
    LAST_CHECKPOINT,
    LOGITS,
    MAP,
    MASKS,
    MAX,
    MIN,
    MODEL_CHECKPOINT,
    MULTICLASS,
    NER,
    NER_RET,
    NUMERICAL,
    OBJECT_DETECTION,
    OCR_TEXT_DETECTION,
    OCR_TEXT_RECOGNITION,
    OVERALL_F1,
    RAY_TUNE_CHECKPOINT,
    REGRESSION,
    ROIS,
    SCORE,
    TEXT,
    TEXT_NER,
    UNIFORM_SOUP,
    XYWH,
    Y_PRED,
    Y_PRED_PROB,
    Y_TRUE,
    ZERO_SHOT_IMAGE_CLASSIFICATION,
)
from autogluon.multimodal.data.datamodule import BaseDataModule
from autogluon.multimodal.data.infer_types import (
    infer_column_types,
    infer_label_column_type_by_problem_type,
    infer_problem_type_output_shape,
    infer_rois_column_type,
    is_image_column,
)
from autogluon.multimodal.data.preprocess_dataframe import MultiModalFeaturePreprocessor
from autogluon.multimodal.matcher import MultiModalMatcher
from autogluon.multimodal.models.utils import get_model_postprocess_fn
from autogluon.multimodal.optimization.lit_distiller import DistillerLitModule
from autogluon.multimodal.optimization.lit_mmdet import MMDetLitModule
from autogluon.multimodal.optimization.lit_module import LitModule
from autogluon.multimodal.optimization.lit_ner import NerLitModule
from autogluon.multimodal.optimization.losses import RKDLoss
from autogluon.multimodal.optimization.utils import (
    get_loss_func,
    get_metric,
    get_norm_layer_param_names,
    get_trainable_params_efficient_finetune,
)
from autogluon.multimodal.problem_types import PROBLEM_TYPES_REG
from autogluon.multimodal.utils import (
    AutoMMModelCheckpoint,
    AutoMMModelCheckpointIO,
    CustomUnpickler,
    DDPCacheWriter,
    ExportMixin,
    LogFilter,
    apply_log_filter,
    assign_feature_column_names,
    average_checkpoints,
    check_if_packages_installed,
    compute_grad_steps,
    compute_num_gpus,
    compute_score,
    convert_pred_to_xywh,
    create_fusion_data_processors,
    create_fusion_model,
    data_to_df,
    evaluate_coco,
    extract_from_output,
    filter_hyperparameters,
    get_available_devices,
    get_config,
    get_detection_classes,
    get_fit_complete_message,
    get_fit_start_message,
    get_local_pretrained_config_paths,
    get_minmax_mode,
    get_mixup,
    get_stopping_threshold,
    hyperparameter_tune,
    infer_dtypes_by_model_names,
    infer_metrics,
    infer_precision,
    infer_scarcity_mode_by_data_size,
    init_df_preprocessor,
    init_pretrained,
    list_timm_models,
    load_text_tokenizers,
    logits_to_prob,
    merge_bio_format,
    modify_duplicate_model_names,
    object_detection_data_to_df,
    predict,
    process_batch,
    save_pretrained_model_configs,
    save_result_df,
    save_text_tokenizers,
    select_model,
    setup_detection_train_tuning_data,
    setup_save_path,
    split_hyperparameters,
    tensor_to_ndarray,
    try_to_infer_pos_label,
    turn_on_off_feature_column_info,
    update_config_by_rules,
    update_hyperparameters,
    update_tabular_config_by_resources,
    upgrade_config,
)

logger = logging.getLogger(__name__)


class DefaultLearner(AbstractMMLearner):
    def __init__(
        self,
        label: Optional[str] = None,
        problem_type: Optional[str] = None,
        column_types: Optional[dict] = None,
        # query: Optional[Union[str, List[str]]] = None,
        # response: Optional[Union[str, List[str]]] = None,
        # match_label: Optional[Union[int, str]] = None,
        # pipeline: Optional[str] = None,
        presets: Optional[str] = None,
        eval_metric: Optional[str] = None,
        path: Optional[str] = None,
        verbosity: Optional[int] = 2,
        num_classes: Optional[int] = None,
        classes: Optional[list] = None,
        enable_progress_bar: Optional[bool] = None,
    ):
        super().__init__()

        self._label_column = label
        self._problem_type = problem_type
        self._presets = presets.lower() if presets else None
        self._eval_metric_name = eval_metric
        self._validation_metric_name = None
        self._output_shape = num_classes
        self._classes = classes  # TODO: To be refactored into detection learner only

        self._ckpt_path = None
        self._pretrained_path = None
        self._config = None
        self._df_preprocessor = None  # initialize it as None.
        self._column_types = column_types  # predictor will setup column_types, and set values here.
        self._data_processors = None  # initialize it as None.
        self._model_postprocess_fn = None
        self._model = None
        # self._resume = False  # This is for predictor to handle.
        self._verbosity = verbosity
        # self._warn_if_exist = warn_if_exist  # unused
        self._enable_progress_bar = enable_progress_bar if enable_progress_bar is not None else True
        # self._init_scratch = init_scratch  # NOTE: is this for predictor to handle?
        # self._sample_data_path = sample_data_path  # this is for detection problem only
        self._fit_called = False  # While using ddp, after fit called, we can only use single gpu.
        self._save_path = path

        # Summary statistics used in fit summary. TODO: wrap it in a class.
        # self._total_train_time = None
        self._best_score = None

    def fit(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        validation_metric_name: str,
        minmax_mode: str,  # this is determined solely on validation_metric_name
        max_time: timedelta,
        save_path: str,
        ckpt_path: str,  # these two are synced
        resume: bool,  # these two are synced
        enable_progress_bar: bool,  # can be inferred?
        presets: Optional[str] = None,
        config: Optional[dict] = None,
        hyperparameters: Optional[Union[str, Dict, List[str]]] = None,
        advanced_hyperparameters: Optional[Dict] = None,
        teacher_predictor: Union[str, MultiModalPredictor] = None,
        hpo_mode: bool = False,
        standalone: bool = True,
        clean_ckpts: bool = True,
        **hpo_kwargs,
    ):
        # TODO(?) We should have a separate "_pre_training_event()" for logging messages.
        logger.info(get_fit_start_message(save_path, validation_metric_name))

        ### SETUP Environments ###
        # 1. get config.
        config, df_preprocessor, grad_steps, strategy, use_ray_lightning = self._setup_environment(
            train_df, presets, hyperparameters, teacher_predictor, hpo_mode, **hpo_kwargs
        )

        # 4. if NER, update output shape. NOTE: This can be refactored into the NER Learner
        # Update output_shape with label_generator.
        model = self._create_model(config, df_preprocessor)

        # 6. get trainable layer params for efficient finetuning only?
        trainable_param_names = self._get_trainable_params(config, model)

        # 7. get data_processors.
        data_processors = self._get_data_processors(config, advanced_hyperparameters, model)

        # 8. infer positive labels
        pos_label = try_to_infer_pos_label(
            data_config=config.data,
            label_encoder=df_preprocessor.label_generator,
            problem_type=self._problem_type,
        )

        # 9. setup validation metric
        validation_metric, custom_metric_func = self._setup_validation_metric(validation_metric_name, pos_label)

        # 10. setup mix up if applicable
        mixup_active, mixup_fn = self._setup_mixup(config)

        # 11. get loss function, NOTE: This can be refactored into Learner class
        loss_func = get_loss_func(
            problem_type=self._problem_type,
            mixup_active=mixup_active,
            loss_func_name=OmegaConf.select(config, "optimization.loss_function"),
            config=config.optimization,
        )

        # 12. get post process function
        model_postprocess_fn = get_model_postprocess_fn(
            problem_type=self._problem_type,
            loss_func=loss_func,
        )

        # 13. assign properties
        self._config = config
        self._df_preprocessor = df_preprocessor
        self._data_processors = data_processors
        self._model = model
        self._model_postprocess_fn = model_postprocess_fn

        # TODO: complete this later
        # 14. if times up. return final model
        if max_time == timedelta(seconds=0):
            # self._top_k_average(
            #     model=model,
            #     save_path=save_path,
            #     minmax_mode=minmax_mode,
            #     is_distill=False,
            #     top_k_average_method=config.optimization.top_k_average_method,
            #     val_df=val_df,
            #     validation_metric_name=validation_metric_name,
            #     strict_loading=not trainable_param_names,
            #     standalone=standalone,
            #     clean_ckpts=clean_ckpts,
            # )

            return self

        # By Default, no distillation. So ignore for now.
        # TODO: complete this later. This can be wrapped into another function called learner.distill()
        # 15. setup distillation.
        # need to assign the above attributes before setting up distillation
        # if teacher_predictor is not None:
        #     (
        #         teacher_model,
        #         critics,
        #         baseline_funcs,
        #         soft_label_loss_func,
        #         softmax_regression_loss_func,
        #         output_feature_adaptor,
        #         output_feature_loss_func,
        #         rkd_loss_func,
        #         teacher_df_preprocessor,
        #         teacher_data_processors,
        #     ) = self._setup_distillation(
        #         teacher_predictor=teacher_predictor,
        #     )
        # else:
        #     (
        #         teacher_model,
        #         critics,
        #         baseline_funcs,
        #         soft_label_loss_func,
        #         softmax_regression_loss_func,
        #         output_feature_adaptor,
        #         output_feature_loss_func,
        #         rkd_loss_func,
        #         teacher_df_preprocessor,
        #         teacher_data_processors,
        #     ) = (None, None, None, None, None, None, None, None, None, None)

        # if teacher_df_preprocessor is not None:
        #     df_preprocessor = [df_preprocessor, teacher_df_preprocessor]
        # if teacher_data_processors is not None:
        #     data_processors = [data_processors, teacher_data_processors]

        val_use_training_mode = (self._problem_type == OBJECT_DETECTION) and (validation_metric_name != MAP)

        # 16. setup pl training data module
        train_dm = self._get_train_data_module(
            train_df, val_df, config, df_preprocessor, data_processors, val_use_training_mode
        )
        # 17. setup optim args
        optimization_kwargs = self._get_optim_kwargs(config)
        # 18. setup metrics args
        metrics_kwargs = self._get_metrics_kwargs(validation_metric_name, validation_metric, custom_metric_func)

        # By default, we use the default task type a.k.a. LitModule
        ## TODO: complete this later in different learners
        # 19. select the correct pl module for a given problem type.
        # NOTE: The task creation and all related variables/functions should be refactored into each Learner class
        # is_distill = teacher_model is not None
        # if is_distill:
        #     output_feature_loss_weight = OmegaConf.select(
        #         self._config, "distiller.output_feature_loss_weight", default=0.0
        #     )
        #     softmax_regression_weight = OmegaConf.select(
        #         self._config, "distiller.softmax_regression_weight", default=0.0
        #     )
        #     use_raw_features = OmegaConf.select(self._config, "distiller.use_raw_features", default=False)
        #     task = DistillerLitModule(
        #         student_model=model,
        #         teacher_model=teacher_model,
        #         matches=config.distiller.matches,
        #         critics=critics,
        #         baseline_funcs=baseline_funcs,
        #         hard_label_weight=config.distiller.hard_label_weight,
        #         soft_label_weight=config.distiller.soft_label_weight,
        #         softmax_regression_weight=softmax_regression_weight,
        #         temperature=config.distiller.temperature,
        #         output_feature_loss_weight=output_feature_loss_weight,
        #         hard_label_loss_func=loss_func,
        #         soft_label_loss_func=soft_label_loss_func,
        #         softmax_regression_loss_func=softmax_regression_loss_func,
        #         output_feature_adaptor=output_feature_adaptor,
        #         output_feature_loss_func=output_feature_loss_func,
        #         rkd_loss_func=rkd_loss_func,
        #         **metrics_kwargs,
        #         **optimization_kwargs,
        #     )
        # elif self._problem_type == NER:
        #     task = NerLitModule(
        #         model=model,
        #         loss_func=loss_func,
        #         efficient_finetune=OmegaConf.select(config, "optimization.efficient_finetune"),
        #         mixup_fn=mixup_fn,
        #         mixup_off_epoch=OmegaConf.select(config, "data.mixup.turn_off_epoch"),
        #         model_postprocess_fn=model_postprocess_fn,
        #         trainable_param_names=trainable_param_names,
        #         **metrics_kwargs,
        #         **optimization_kwargs,
        #     )
        # elif self._problem_type == OBJECT_DETECTION:
        #     task = MMDetLitModule(
        #         model=model,
        #         **metrics_kwargs,
        #         **optimization_kwargs,
        #     )
        # else:
        #     task = LitModule(
        #         model=model,
        #         loss_func=loss_func,
        #         efficient_finetune=OmegaConf.select(config, "optimization.efficient_finetune"),
        #         mixup_fn=mixup_fn,
        #         mixup_off_epoch=OmegaConf.select(config, "data.mixup.turn_off_epoch"),
        #         model_postprocess_fn=model_postprocess_fn,
        #         trainable_param_names=trainable_param_names,
        #         skip_final_val=OmegaConf.select(config, "optimization.skip_final_val", default=False),
        #         **metrics_kwargs,
        #         **optimization_kwargs,
        #     )

        task = self._create_task(
            config,
            model,
            trainable_param_names,
            mixup_fn,
            loss_func,
            model_postprocess_fn,
            optimization_kwargs,
            metrics_kwargs,
        )

        # 20. Setup call backs
        checkpoint_callback = self._create_checkpoint_callback(minmax_mode, save_path, config, task)
        early_stopping_callback = self._create_early_stop_callback(validation_metric_name, minmax_mode, config, task)
        lr_callback = self._create_lr_callback()
        model_summary = pl.callbacks.ModelSummary(max_depth=1)
        callbacks = [
            checkpoint_callback,
            early_stopping_callback,
            lr_callback,
            model_summary,
        ]

        # 21. for hpo with ray
        # TODO: Leave this out for now. We will add this back later
        # use_ray_lightning = "_ray_lightning_plugin" in hpo_kwargs
        # if hpo_mode:
        #     if use_ray_lightning:
        #         from ray_lightning.tune import TuneReportCheckpointCallback
        #     else:
        #         from ray.tune.integration.pytorch_lightning import TuneReportCheckpointCallback
        #     tune_report_callback = TuneReportCheckpointCallback(
        #         {f"{task.validation_metric_name}": f"{task.validation_metric_name}"},
        #         filename=RAY_TUNE_CHECKPOINT,
        #     )
        #     callbacks = [
        #         tune_report_callback,
        #         early_stopping_callback,
        #         lr_callback,
        #         model_summary,
        #     ]

        custom_checkpoint_plugin = AutoMMModelCheckpointIO(
            trainable_param_names=trainable_param_names,
            model_name_to_id=model.name_to_id,
        )

        tb_logger = pl.loggers.TensorBoardLogger(
            save_dir=save_path,
            name="",
            version="",
        )

        ## Moved to the begining to join the setup steps for config
        # # 22. training environment-dependent configs
        # num_gpus = compute_num_gpus(config_num_gpus=config.env.num_gpus, strategy=config.env.strategy)

        # precision = infer_precision(num_gpus=num_gpus, precision=config.env.precision)

        # if num_gpus == 0:  # CPU only training
        #     grad_steps = max(
        #         config.env.batch_size // (config.env.per_gpu_batch_size * config.env.num_nodes),
        #         1,
        #     )
        # else:
        #     grad_steps = max(
        #         config.env.batch_size // (config.env.per_gpu_batch_size * num_gpus * config.env.num_nodes),
        #         1,
        #     )

        # if not hpo_mode:
        #     if num_gpus <= 1:
        #         if config.env.strategy == DEEPSPEED_OFFLOADING:  # Offloading currently only tested for single GPU
        #             assert version.parse(pl.__version__) >= version.parse(
        #                 DEEPSPEED_MIN_PL_VERSION
        #             ), f"For DeepSpeed Offloading to work reliably you need at least pytorch-lightning version {DEEPSPEED_MIN_PL_VERSION}, however, found {pl.__version__}. Please update your pytorch-lightning version."
        #             from .optimization.deepspeed import CustomDeepSpeedStrategy

        #             strategy = CustomDeepSpeedStrategy(
        #                 stage=3,
        #                 offload_optimizer=True,
        #                 offload_parameters=False,
        #                 allgather_bucket_size=config.env.deepspeed_allgather_size,
        #                 reduce_bucket_size=config.env.deepspeed_allreduce_size,
        #             )
        #         else:
        #             strategy = None
        #     else:
        #         strategy = config.env.strategy
        # else:
        #     # we don't support running each trial in parallel without ray lightning
        #     if use_ray_lightning:
        #         strategy = hpo_kwargs.get("_ray_lightning_plugin")
        #     else:
        #         strategy = None
        #         num_gpus = min(num_gpus, 1)

        # config.env.num_gpus = num_gpus
        # config.env.precision = precision
        # config.env.strategy = strategy if not config.env.strategy == DEEPSPEED_OFFLOADING else DEEPSPEED_OFFLOADING
        # self._config = config

        ## IGNORE THIS FOR NOW
        # save artifacts for the current running, except for model checkpoint, which will be saved in trainer
        # self.save(save_path, standalone=standalone)

        blacklist_msgs = ["already configured with model summary"]
        log_filter = LogFilter(blacklist_msgs)
        # 23. Declare pl.Trainer
        with apply_log_filter(log_filter):
            trainer = pl.Trainer(
                accelerator="gpu" if config.env.num_gpus > 0 else None,
                devices=get_available_devices(
                    num_gpus=config.env.num_gpus,
                    auto_select_gpus=config.env.auto_select_gpus,
                    use_ray_lightning=use_ray_lightning,
                ),
                num_nodes=config.env.num_nodes,
                precision=config.env.precision,
                strategy=strategy,
                benchmark=False,
                deterministic=config.env.deterministic,
                max_epochs=config.optimization.max_epochs,
                max_steps=config.optimization.max_steps,
                max_time=max_time,
                callbacks=callbacks,
                logger=tb_logger,
                gradient_clip_val=OmegaConf.select(config, "optimization.gradient_clip_val", default=1),
                gradient_clip_algorithm=OmegaConf.select(
                    config, "optimization.gradient_clip_algorithm", default="norm"
                ),
                accumulate_grad_batches=grad_steps,
                log_every_n_steps=OmegaConf.select(config, "optimization.log_every_n_steps", default=10),
                enable_progress_bar=enable_progress_bar,
                fast_dev_run=config.env.fast_dev_run,
                track_grad_norm=OmegaConf.select(config, "optimization.track_grad_norm", default=-1),
                val_check_interval=config.optimization.val_check_interval,
                check_val_every_n_epoch=config.optimization.check_val_every_n_epoch
                if hasattr(config.optimization, "check_val_every_n_epoch")
                else 1,
                plugins=[custom_checkpoint_plugin],
            )

        # 24. pl.Trainer.fit()
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                ".*does not have many workers which may be a bottleneck. "
                "Consider increasing the value of the `num_workers` argument` "
                ".* in the `DataLoader` init to improve performance.*",
            )
            warnings.filterwarnings("ignore", "Checkpoint directory .* exists and is not empty.")
            trainer.fit(
                task,
                datamodule=train_dm,
                ckpt_path=ckpt_path if resume else None,  # this is to resume training that was broken accidentally
            )

        # 25. execute call bakcs at training finish
        if trainer.global_rank == 0:
            # We do not perform averaging checkpoint in the case of hpo for each trial
            # We only averaging the checkpoint of the best trial in the end in the master process
            # if not hpo_mode:
            # self._top_k_average(
            #     model=model,
            #     save_path=save_path,
            #     minmax_mode=minmax_mode,
            #     is_distill=False,  # By default, we do not do distill. If distill, we will do it in the distill learner
            #     top_k_average_method=config.optimization.top_k_average_method,
            #     val_df=val_df,
            #     validation_metric_name=validation_metric_name,
            #     strategy=strategy,
            #     strict_loading=not trainable_param_names,
            #     # Not strict loading if using parameter-efficient finetuning
            #     standalone=standalone,
            #     clean_ckpts=clean_ckpts,
            # )
            self._best_score = trainer.callback_metrics[f"val_{self._validation_metric_name}"].item()
        else:
            sys.exit(f"Training finished, exit the process with global_rank={trainer.global_rank}...")

    def _top_k_average(
        self,
        model,
        save_path,
        minmax_mode,
        is_distill,
        top_k_average_method,
        val_df,
        validation_metric_name,
        strategy=None,
        last_ckpt_path=None,
        strict_loading=True,
        standalone=True,
        clean_ckpts=True,
    ):
        # FIXME: we need to change validation_metric to evaluation_metric for model choosing
        # since we called self.evaluate. Below is a temporal fix for NER.
        if self._problem_type is not None and self._problem_type == NER:
            validation_metric_name = OVERALL_F1  # seqeval only support overall_f1

        best_k_models_yaml_path = os.path.join(save_path, BEST_K_MODELS_FILE)
        if os.path.exists(best_k_models_yaml_path):
            with open(best_k_models_yaml_path, "r") as f:
                best_k_models = yaml.safe_load(f)

        else:
            # In some cases, the training ends up too early (e.g., due to time_limit) so that there is
            # no saved best_k model checkpoints. In that scenario, we won't perform any model averaging.
            best_k_models = None
        if last_ckpt_path is None:
            last_ckpt_path = os.path.join(save_path, LAST_CHECKPOINT)

        if is_distill:
            prefix = "student_model."
        else:
            prefix = "model."

        if best_k_models:
            if top_k_average_method == UNIFORM_SOUP:
                logger.info(f"Start to fuse {len(best_k_models)} checkpoints via the uniform soup algorithm.")
                ingredients = top_k_model_paths = list(best_k_models.keys())
            else:
                top_k_model_paths = [
                    v[0]
                    for v in sorted(
                        list(best_k_models.items()),
                        key=lambda ele: ele[1],
                        reverse=(minmax_mode == MAX),
                    )
                ]
                if top_k_average_method == GREEDY_SOUP:
                    # Select the ingredients based on the methods proposed in paper
                    #  "Model soups: averaging weights of multiple fine-tuned models improves accuracy without
                    #  increasing inference time", https://arxiv.org/pdf/2203.05482.pdf
                    monitor_op = {MIN: operator.le, MAX: operator.ge}[minmax_mode]
                    ingredients = [top_k_model_paths[0]]
                    if len(top_k_model_paths) > 1:
                        logger.info(
                            f"Start to fuse {len(top_k_model_paths)} checkpoints via the greedy soup algorithm."
                        )

                        self._model = self._load_state_dict(
                            model=model,
                            path=top_k_model_paths[0],
                            prefix=prefix,
                            strict=strict_loading,
                        )
                        best_score = self.evaluate(val_df, metrics=[validation_metric_name])[validation_metric_name]
                        for i in range(1, len(top_k_model_paths)):
                            cand_avg_state_dict = average_checkpoints(
                                checkpoint_paths=ingredients + [top_k_model_paths[i]],
                            )
                            self._model = self._load_state_dict(
                                model=self._model,
                                state_dict=cand_avg_state_dict,
                                prefix=prefix,
                                strict=strict_loading,
                            )
                            cand_score = self.evaluate(val_df, metrics=[validation_metric_name])[
                                validation_metric_name
                            ]
                            if monitor_op(cand_score, best_score):
                                # Add new ingredient
                                ingredients.append(top_k_model_paths[i])
                                best_score = cand_score
                elif top_k_average_method == BEST:
                    ingredients = [top_k_model_paths[0]]
                else:
                    raise ValueError(
                        f"The key for 'optimization.top_k_average_method' is not supported. "
                        f"We only support '{GREEDY_SOUP}', '{UNIFORM_SOUP}' and '{BEST}'. "
                        f"The provided value is '{top_k_average_method}'."
                    )
        else:
            # best_k_models is empty so we will manually save a checkpoint from the trainer
            # and use it as the main ingredients
            ingredients = [last_ckpt_path]
            top_k_model_paths = []
            # no checkpoints are available, do nothing
            if not os.path.isfile(last_ckpt_path):
                return

        # Average all the ingredients
        avg_state_dict = average_checkpoints(
            checkpoint_paths=ingredients,
        )
        self._model = self._load_state_dict(
            model=model,
            state_dict=avg_state_dict,
            prefix=prefix,
            strict=strict_loading,
        )

        if is_distill:
            avg_state_dict = self._replace_model_name_prefix(
                state_dict=avg_state_dict,
                old_prefix="student_model",
                new_prefix="model",
            )

        if not standalone:
            checkpoint = {"state_dict": avg_state_dict}
        else:
            if strategy and hasattr(strategy, "strategy_name") and strategy.strategy_name == DEEPSPEED_STRATEGY:
                checkpoint = {
                    "state_dict": {
                        name.partition("module.")[2]: param
                        for name, param in strategy.model._zero3_consolidated_16bit_state_dict().items()
                    }
                }
            else:
                checkpoint = {
                    "state_dict": {"model." + name: param for name, param in self._model.state_dict().items()}
                }

        torch.save(checkpoint, os.path.join(save_path, MODEL_CHECKPOINT))

        if clean_ckpts:
            # clean old checkpoints + the intermediate files stored
            for per_path in top_k_model_paths:
                if os.path.isfile(per_path):
                    os.remove(per_path)
            # remove the yaml file after cleaning the checkpoints
            if os.path.isfile(best_k_models_yaml_path):
                os.remove(best_k_models_yaml_path)
            # clean the last checkpoint
            if os.path.isfile(last_ckpt_path):
                os.remove(last_ckpt_path)

    def _create_lr_callback(self):
        """Creates the learning rate callback
        By default, we use the pl LearningRateMonitor callback
        This can be overriden by child learners to use other learning rate callbacks

        Returns:
            _type_: _description_
        """
        lr_callback = pl.callbacks.LearningRateMonitor(logging_interval="step")
        return lr_callback

    def _create_early_stop_callback(self, validation_metric_name, minmax_mode, config, task):
        """Creates the callback for early stopping during training
        By default, we use the pl EarlyStopping callback
        This can be overriden by child learners to use other early stopping callbacks

        Args:
            validation_metric_name (_type_): _description_
            minmax_mode (_type_): _description_
            config (_type_): _description_
            task (_type_): _description_

        Returns:
            _type_: _description_
        """
        early_stopping_callback = pl.callbacks.EarlyStopping(
            monitor=task.validation_metric_name,
            patience=config.optimization.patience,
            mode=minmax_mode,
            stopping_threshold=get_stopping_threshold(validation_metric_name),
        )

        return early_stopping_callback

    def _create_checkpoint_callback(self, minmax_mode, save_path, config, task):
        """Creates the pl checkpoint callback during training
        By default, we use the AutoMMModelCheckpoint, which is a customized checkpoint callback
        This can be overriden by child learners to use other checkpoint callbacks

        Args:
            minmax_mode (_type_): _description_
            save_path (_type_): _description_
            config (_type_): _description_
            task (_type_): _description_

        Returns:
            _type_: _description_
        """
        checkpoint_callback = AutoMMModelCheckpoint(
            dirpath=save_path,
            save_top_k=config.optimization.top_k,
            verbose=True,
            monitor=task.validation_metric_name,
            mode=minmax_mode,
            save_last=True,
        )

        return checkpoint_callback

    def _create_task(
        self,
        config,
        model,
        trainable_param_names,
        mixup_fn,
        loss_func,
        model_postprocess_fn,
        optimization_kwargs,
        metrics_kwargs,
    ) -> pl.LightningModule:
        """
        Select the correct pl task module for a given problem type.
        By default, we use the default task type a.k.a. LitModule
        This can be overridden by a child learner if needed.


        Args:
            config (_type_): _description_
            model (_type_): _description_
            trainable_param_names (_type_): _description_
            mixup_fn (_type_): _description_
            loss_func (_type_): _description_
            model_postprocess_fn (_type_): _description_
            optimization_kwargs (_type_): _description_
            metrics_kwargs (_type_): _description_

        Returns:
            _type_: _description_
        """
        task = LitModule(
            model=model,
            loss_func=loss_func,
            efficient_finetune=OmegaConf.select(config, "optimization.efficient_finetune"),
            mixup_fn=mixup_fn,
            mixup_off_epoch=OmegaConf.select(config, "data.mixup.turn_off_epoch"),
            model_postprocess_fn=model_postprocess_fn,
            trainable_param_names=trainable_param_names,
            skip_final_val=OmegaConf.select(config, "optimization.skip_final_val", default=False),
            **metrics_kwargs,
            **optimization_kwargs,
        )

        return task

    def _get_metrics_kwargs(self, validation_metric_name, validation_metric, custom_metric_func) -> dict:
        metrics_kwargs = dict(
            validation_metric=validation_metric,
            validation_metric_name=validation_metric_name,
            custom_metric_func=custom_metric_func,
        )

        return metrics_kwargs

    def _get_optim_kwargs(self, config) -> dict:
        optimization_kwargs = dict(
            optim_type=config.optimization.optim_type,
            lr_choice=config.optimization.lr_choice,
            lr_schedule=config.optimization.lr_schedule,
            lr=config.optimization.learning_rate,
            lr_decay=config.optimization.lr_decay,
            end_lr=config.optimization.end_lr,
            lr_mult=config.optimization.lr_mult,
            weight_decay=config.optimization.weight_decay,
            warmup_steps=config.optimization.warmup_steps,
        )

        return optimization_kwargs

    def _get_train_data_module(
        self, train_df, val_df, config, df_preprocessor, data_processors, val_use_training_mode
    ) -> LightningDataModule:
        """get the data module for training
        By default, we use the default data module a.k.a. BaseDataModule
        This is to be overridden by a child learner if needed.

        Args:
            train_df (_type_): _description_
            val_df (_type_): _description_
            config (_type_): _description_
            df_preprocessor (_type_): _description_
            data_processors (_type_): _description_
            val_use_training_mode (_type_): _description_

        Returns:
            _type_: _description_
        """
        train_dm = BaseDataModule(
            df_preprocessor=df_preprocessor,
            data_processors=data_processors,
            per_gpu_batch_size=config.env.per_gpu_batch_size,
            num_workers=config.env.num_workers,
            train_data=train_df,
            validate_data=val_df,
            val_use_training_mode=val_use_training_mode,
        )

        return train_dm

    def _setup_mixup(self, config):
        mixup_active, mixup_fn = get_mixup(
            model_config=OmegaConf.select(config, "model"),
            mixup_config=OmegaConf.select(config, "data.mixup"),
            num_classes=self._output_shape,
        )
        if mixup_active and (config.env.per_gpu_batch_size == 1 or config.env.per_gpu_batch_size % 2 == 1):
            warnings.warn(
                "The mixup is done on the batch."
                "The per_gpu_batch_size should be >1 and even for reasonable operation",
                UserWarning,
            )

        return mixup_active, mixup_fn

    def _setup_validation_metric(self, validation_metric_name, pos_label):
        if validation_metric_name is not None:
            validation_metric, custom_metric_func = get_metric(
                metric_name=validation_metric_name,
                num_classes=self._output_shape,
                pos_label=pos_label,
            )
        else:
            validation_metric, custom_metric_func = (None, None)
        return validation_metric, custom_metric_func

    def _get_data_processors(self, config, advanced_hyperparameters, model):
        if self._data_processors is None:
            data_processors = create_fusion_data_processors(
                config=config,
                model=model,
                advanced_hyperparameters=advanced_hyperparameters,
            )
        else:  # continuing training
            data_processors = self._data_processors
        return data_processors

    def _get_trainable_params(self, config, model):
        norm_param_names = get_norm_layer_param_names(model)

        trainable_param_names = get_trainable_params_efficient_finetune(
            norm_param_names,
            efficient_finetune=OmegaConf.select(config, "optimization.efficient_finetune"),
        )

        return trainable_param_names

    def _create_model(self, config, df_preprocessor) -> nn.Module:
        """Creates the model for training
        By default we use the create_fusion_model function to create the model
        This can be overriden by a child learner if needed.

        Args:
            config (_type_): _description_
            df_preprocessor (_type_): _description_

        Returns:
            nn.Module: _description_
        """
        ##  this is to be moved into NER learners
        if self._problem_type == NER:
            self._output_shape = len(df_preprocessor.label_generator.unique_entity_groups)

        # 5. setup model.
        # create a new model if calling ._fit() for the first time. otherwise re-use the existing self._model
        if self._model is None:
            model = create_fusion_model(
                config=config,
                num_classes=self._output_shape,
                classes=self._classes,
                num_numerical_columns=len(df_preprocessor.numerical_feature_names),
                num_categories=df_preprocessor.categorical_num_categories,
            )
        else:  # continuing training
            model = self._model
        return model

    def _setup_environment(
        self,
        train_df: pd.DataFrame,
        presets: Optional[str],
        hyperparameters: Optional[Union[str, List[str], Dict]],
        teacher_predictor: Union[str, MultiModalPredictor],
        hpo_mode: bool,
        **hpo_kwargs,
    ):
        """Set up training environment parameters: config, df_preprocessor, grad_step, strategy, etc.
        TODO: Can df_preprocessor be moved outside this function?

        Args:
            train_df (pd.DataFrame): training data frame
            presets (Optional[str]): preset to use
            hyperparameters (Optional[Union[str, List[str], Dict]]): user provided hyperparameter overrides
            teacher_predictor (Union[str, MultiModalPredictor]): teacher model for knowledge distillation
            hpo_mode (_type_): if running hpo
            hpo_kwargs (_type_): hpo params

        Returns:
            config: dict
            df_preprocessor: MultiModalFeaturePreprocessor
            grad_step: int
            strategy: str
        """
        config = get_config(
            problem_type=self._problem_type,
            presets=presets,
            config=config,
            overrides=hyperparameters,
            extra=["distiller"] if teacher_predictor is not None else None,
        )

        config = update_config_by_rules(
            problem_type=self._problem_type,
            config=config,
        )

        # 2. set up df_preprocessor if not given yet
        if self._df_preprocessor is None:
            df_preprocessor = init_df_preprocessor(
                config=config,
                column_types=self._column_types,
                label_column=self._label_column,
                train_df_x=train_df.drop(columns=self._label_column),
                train_df_y=train_df[self._label_column],
            )
        else:  # continuing training
            df_preprocessor = self._df_preprocessor

        # 3. update config given df_preprocessor
        # Avoid passing tabular data with many columns to MultiHeadAttention.
        # If models have additive_attention="auto", we enable it automatically for large tables.

        # NOTE: This part only changes config.model and config.env
        config = update_tabular_config_by_resources(
            config,
            num_numerical_columns=len(df_preprocessor.numerical_feature_names),
            num_categorical_columns=len(df_preprocessor.categorical_num_categories),
        )

        # This part only changes config.model
        config = select_model(config=config, df_preprocessor=df_preprocessor)

        # 22. training environment-dependent configs
        num_gpus = compute_num_gpus(config_num_gpus=config.env.num_gpus, strategy=config.env.strategy)

        precision = infer_precision(num_gpus=num_gpus, precision=config.env.precision)

        grad_steps = compute_grad_steps(config=config, num_gpus=num_gpus)

        use_ray_lightning = "_ray_lightning_plugin" in hpo_kwargs
        num_gpus, strategy = self._compute_strategy(config, hpo_mode, hpo_kwargs, num_gpus, use_ray_lightning)

        config.env.num_gpus = num_gpus
        config.env.precision = precision
        config.env.strategy = strategy if not config.env.strategy == DEEPSPEED_OFFLOADING else DEEPSPEED_OFFLOADING

        self._config = config
        return config, df_preprocessor, grad_steps, strategy, use_ray_lightning

    def _compute_strategy(self, config, hpo_mode, hpo_kwargs, num_gpus, use_ray_lightning):
        if not hpo_mode:
            if num_gpus <= 1:
                if config.env.strategy == DEEPSPEED_OFFLOADING:  # Offloading currently only tested for single GPU
                    assert version.parse(pl.__version__) >= version.parse(
                        DEEPSPEED_MIN_PL_VERSION
                    ), f"For DeepSpeed Offloading to work reliably you need at least pytorch-lightning version {DEEPSPEED_MIN_PL_VERSION}, however, found {pl.__version__}. Please update your pytorch-lightning version."
                    from autogluon.multimodal.optimization.deepspeed import CustomDeepSpeedStrategy

                    strategy = CustomDeepSpeedStrategy(
                        stage=3,
                        offload_optimizer=True,
                        offload_parameters=False,
                        allgather_bucket_size=config.env.deepspeed_allgather_size,
                        reduce_bucket_size=config.env.deepspeed_allreduce_size,
                    )
                else:
                    strategy = None
            else:
                strategy = config.env.strategy
        else:
            # we don't support running each trial in parallel without ray lightning
            if use_ray_lightning:
                strategy = hpo_kwargs.get("_ray_lightning_plugin")
            else:
                strategy = None
                num_gpus = min(num_gpus, 1)
        return num_gpus, strategy