import json
import logging
import os
from datetime import timedelta
from typing import Dict, List, Optional, Union

import pandas as pd
from omegaconf import DictConfig, OmegaConf
from torch import nn

from ..constants import (
    BBOX,
    DDP,
    MAP,
    MULTI_IMAGE_MIX_DATASET,
    OBJECT_DETECTION,
    OPEN_VOCABULARY_OBJECT_DETECTION,
    OVD_RET,
    XYWH,
)
from ..data import BaseDataModule, MultiImageMixDataset, MultiModalFeaturePreprocessor, infer_rois_column_type
from ..optimization import LitModule, MMDetLitModule
from ..utils import (
    check_if_packages_installed,
    cocoeval,
    convert_pred_to_xywh,
    create_fusion_model,
    extract_from_output,
    from_coco_or_voc,
    get_detection_classes,
    object_detection_data_to_df,
    save_ovd_result_df,
    save_result_df,
    setup_save_path,
    split_train_tuning_data,
)
from .base import BaseLearner

logger = logging.getLogger(__name__)


class ObjectDetectionLearner(BaseLearner):
    def __init__(
        self,
        label: Optional[str] = None,  # TODO: can we let users customize label?
        problem_type: Optional[str] = OBJECT_DETECTION,
        presets: Optional[str] = None,
        eval_metric: Optional[str] = None,
        hyperparameters: Optional[dict] = None,
        path: Optional[str] = None,
        verbosity: Optional[int] = 2,
        num_classes: Optional[int] = None,  # TODO: can we infer this from train/predict data?
        classes: Optional[list] = None,  # TODO: can we infer this from train/predict data?
        warn_if_exist: Optional[bool] = True,
        enable_progress_bar: Optional[bool] = None,
        pretrained: Optional[bool] = True,
        validation_metric: Optional[str] = None,
        sample_data_path: Optional[str] = None,  # TODO: can we use train/predict data instead?
        **kwargs,
    ):
        """
        Parameters
        ----------
        num_classes
            Number of classes. Used in classification.
            If this is specified and is different from the pretrained model's output,
            the model's head will be changed to have <num_classes> output.
        classes
            All classes in this dataset.
        sample_data_path
            This is used for automatically inference num_classes, classes, or label.
        """
        super().__init__(
            problem_type=problem_type,
            presets=presets,
            eval_metric=eval_metric,
            hyperparameters=hyperparameters,
            path=path,
            verbosity=verbosity,
            warn_if_exist=warn_if_exist,
            enable_progress_bar=enable_progress_bar,
            pretrained=pretrained,
            validation_metric=validation_metric,
        )
        check_if_packages_installed(problem_type=self._problem_type)

        self._output_shape = num_classes
        self._classes = classes
        self._sample_data_path = sample_data_path

        # TODO: merge object detection and open vocabulary object detection
        if self._problem_type == OBJECT_DETECTION:
            self._label_column = "label"
            if self._sample_data_path is not None:
                self._classes = get_detection_classes(self._sample_data_path)
                self._output_shape = len(self._classes)

        # TODO: merge _detection_anno_train and detection_anno_train?
        self._detection_anno_train = None
        self.detection_anno_train = None

    @property
    def classes(self):
        """
        Return the classes of object detection.
        """
        return self._model.model.CLASSES

    def setup_detection_train_tuning_data(self, max_num_tuning_data, seed, train_data, tuning_data):
        if isinstance(train_data, str):
            self._detection_anno_train = train_data
            train_data = from_coco_or_voc(train_data, "train")  # TODO: Refactor to use convert_data_to_df
            if tuning_data is not None:
                self.detection_anno_train = tuning_data
                tuning_data = from_coco_or_voc(tuning_data, "val")  # TODO: Refactor to use convert_data_to_df
                if max_num_tuning_data is not None:
                    if len(tuning_data) > max_num_tuning_data:
                        tuning_data = tuning_data.sample(
                            n=max_num_tuning_data, replace=False, random_state=seed
                        ).reset_index(drop=True)
        elif isinstance(train_data, pd.DataFrame):
            self._detection_anno_train = None
            # sanity check dataframe columns
            train_data = object_detection_data_to_df(train_data)
            if tuning_data is not None:
                self.detection_anno_train = tuning_data
                tuning_data = object_detection_data_to_df(tuning_data)
                if max_num_tuning_data is not None:
                    if len(tuning_data) > max_num_tuning_data:
                        tuning_data = tuning_data.sample(
                            n=max_num_tuning_data, replace=False, random_state=seed
                        ).reset_index(drop=True)
        else:
            raise TypeError(f"Expected train_data to have type str or pd.DataFrame, but got type: {type(train_data)}")
        return train_data, tuning_data

    def prepare_train_tuning_data(
        self,
        train_data: Union[pd.DataFrame, str],
        tuning_data: Optional[Union[pd.DataFrame, str]],
        holdout_frac: Optional[float],
        max_num_tuning_data: Optional[int],
        seed: Optional[int],
    ):
        # TODO: remove self from calling setup_detection_train_tuning_data()
        train_data, tuning_data = self.setup_detection_train_tuning_data(
            train_data=train_data,
            tuning_data=tuning_data,
            max_num_tuning_data=max_num_tuning_data,
            seed=seed,
        )

        if tuning_data is None:
            train_data, tuning_data = split_train_tuning_data(
                data=train_data,
                holdout_frac=holdout_frac,
                problem_type=self._problem_type,
                label_column=self._label_column,
                random_state=seed,
            )

        self._train_data = train_data
        self._tuning_data = tuning_data

    def infer_output_shape(self, **kwargs):
        # TODO: support inferring output during fit()?
        assert self._output_shape is not None, f"output_shape should have been set in the learner initialization."

    def init_pretrained(self):
        super().init_pretrained()
        self._model = create_fusion_model(
            config=self._config,
            pretrained=self._pretrained,
            num_classes=self._output_shape,
            classes=self._classes,
        )

    def fit(
        self,
        train_data: Union[pd.DataFrame, str],
        presets: Optional[str] = None,
        tuning_data: Optional[Union[pd.DataFrame, str]] = None,
        max_num_tuning_data: Optional[int] = None,
        time_limit: Optional[int] = None,
        save_path: Optional[str] = None,
        hyperparameters: Optional[Union[str, Dict, List[str]]] = None,
        column_types: Optional[Dict] = None,
        holdout_frac: Optional[float] = None,
        seed: Optional[int] = 0,
        standalone: Optional[bool] = True,
        hyperparameter_tune_kwargs: Optional[Dict] = None,
        clean_ckpts: Optional[bool] = True,
        **kwargs,
    ):
        training_start = self.on_fit_start(presets=presets)
        self.setup_save_path(save_path=save_path)
        self.prepare_train_tuning_data(
            train_data=train_data,
            tuning_data=tuning_data,
            holdout_frac=holdout_frac,
            max_num_tuning_data=max_num_tuning_data,
            seed=seed,
        )
        self.infer_column_types(column_types=column_types)
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

    def get_datamodule_per_run(
        self,
        df_preprocessor,
        data_processors,
        per_gpu_batch_size,
        num_workers,
        model_config=None,
        predict_data=None,
        is_train=True,
    ):
        datamodule_kwargs = dict(
            df_preprocessor=df_preprocessor,
            data_processors=data_processors,
            per_gpu_batch_size=per_gpu_batch_size,
            num_workers=num_workers,
        )
        if is_train:
            val_use_training_mode = (self._problem_type == OBJECT_DETECTION) and (self._validation_metric_name != MAP)
            datamodule_kwargs.update(
                dict(validate_data=self._tuning_data, val_use_training_mode=val_use_training_mode)
            )
            if (
                self._problem_type == OBJECT_DETECTION
                and model_config is not None
                and MULTI_IMAGE_MIX_DATASET in model_config
            ):
                train_dataset = MultiImageMixDataset(
                    data=self._train_data,
                    preprocessor=[df_preprocessor],
                    processors=[data_processors],
                    model_config=model_config,
                    id_mappings=None,
                    is_training=True,
                )
                datamodule_kwargs.update(dict(train_dataset=train_dataset))
            else:
                datamodule_kwargs.update(dict(train_data=self._train_data))
        else:
            datamodule_kwargs.update(dict(predict_data=predict_data))

        datamodule = BaseDataModule(**datamodule_kwargs)
        return datamodule

    def get_strategy_per_run(self, num_gpus, config):
        if num_gpus <= 1:
            strategy = "auto"
        else:
            strategy = DDP

        return strategy

    def update_num_gpus_by_strategy(self, strategy, num_gpus):
        if strategy == DDP and self._fit_called:
            num_gpus = 1  # While using DDP, we can only use single gpu after fit is called

        return num_gpus

    def get_optimization_kwargs_per_run(self, config, validation_metric, custom_metric_func):
        return dict(
            optim_type=config.optimization.optim_type,
            lr_choice=config.optimization.lr_choice,
            lr_schedule=config.optimization.lr_schedule,
            lr=config.optimization.learning_rate,
            lr_decay=config.optimization.lr_decay,
            end_lr=config.optimization.end_lr,
            lr_mult=config.optimization.lr_mult,
            weight_decay=config.optimization.weight_decay,
            warmup_steps=config.optimization.warmup_steps,
            track_grad_norm=OmegaConf.select(config, "optimization.track_grad_norm", default=-1),
            validation_metric=validation_metric,
            validation_metric_name=self._validation_metric_name,
            custom_metric_func=custom_metric_func,
        )

    def get_litmodule_per_run(
        self,
        model: Optional[nn.Module] = None,
        optimization_kwargs: Optional[dict] = None,
        is_train=True,
    ):
        # add ovd
        if self._problem_type == OPEN_VOCABULARY_OBJECT_DETECTION:
            LightningModule = LitModule
        elif self._problem_type == OBJECT_DETECTION:
            LightningModule = MMDetLitModule
        else:
            raise TypeError(f"problem type {self._problem_type} is not supported by ObjectDetectionLearner.")

        if is_train:
            return LightningModule(
                model=model,
                **optimization_kwargs,
            )
        else:
            return LightningModule(model=self._model)

    def get_model_per_run(self, model, config):
        if model is None:
            model = create_fusion_model(
                config=config,
                num_classes=self._output_shape,
                classes=self._classes,
            )
        return model

    def fit_per_run(
        self,
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
        standalone: bool = True,
        clean_ckpts: bool = True,
    ):
        self.on_fit_per_run_start(seed=seed, save_path=save_path)
        config = self.get_config_per_run(config=config, hyperparameters=hyperparameters)
        df_preprocessor = self.get_df_preprocessor_per_run(
            df_preprocessor=df_preprocessor,
            config=config,
        )
        config = self.update_config_by_data_per_run(config=config, df_preprocessor=df_preprocessor)
        model = self.get_model_per_run(model=model, config=config)
        model = self.compile_model_per_run(config=config, model=model)
        data_processors = self.get_data_processors_per_run(
            data_processors=data_processors,
            config=config,
            model=model,
            advanced_hyperparameters=advanced_hyperparameters,
        )
        validation_metric, custom_metric_func = self.get_validation_metric_per_run()
        if max_time == timedelta(seconds=0):
            return dict(
                config=config,
                df_preprocessor=df_preprocessor,
                data_processors=data_processors,
                model=model,
            )
        datamodule = self.get_datamodule_per_run(
            df_preprocessor=df_preprocessor,
            data_processors=data_processors,
            per_gpu_batch_size=config.env.per_gpu_batch_size,
            num_workers=config.env.num_workers,
            model_config=model.config,
        )
        optimization_kwargs = self.get_optimization_kwargs_per_run(
            config=config,
            validation_metric=validation_metric,
            custom_metric_func=custom_metric_func,
        )
        litmodule = self.get_litmodule_per_run(
            model=model,
            optimization_kwargs=optimization_kwargs,
        )
        callbacks = self.get_callbacks_per_run(save_path=save_path, config=config, litmodule=litmodule)
        plugins = self.get_plugins_per_run(model=model)
        tb_logger = self.get_tb_logger(save_path=save_path)
        num_gpus, strategy = self.get_num_gpus_and_strategy_per_run(config=config)
        precision = self.get_precision_per_run(num_gpus=num_gpus, precision=config.env.precision)
        grad_steps = self.get_grad_steps(num_gpus=num_gpus, config=config)
        strategy = self.get_strategy_per_run(num_gpus=num_gpus, config=config)
        config = self.post_update_config_per_run(
            config=config,
            num_gpus=num_gpus,
            precision=precision,
            strategy=strategy,
        )
        trainer = self.init_trainer_per_run(
            num_gpus=num_gpus,
            config=config,
            precision=precision,
            strategy=strategy,
            max_time=max_time,
            callbacks=callbacks,
            tb_logger=tb_logger,
            grad_steps=grad_steps,
            plugins=plugins,
            enable_progress_bar=enable_progress_bar,
        )

        self.run_trainer(
            trainer=trainer,
            litmodule=litmodule,
            datamodule=datamodule,
            ckpt_path=ckpt_path,
            resume=resume,
        )
        self.on_fit_per_run_end(
            save_path=save_path,
            standalone=standalone,
            trainer=trainer,
            config=config,
            df_preprocessor=df_preprocessor,
            data_processors=data_processors,
            model=model,
        )

        return dict(
            config=config,
            df_preprocessor=df_preprocessor,
            data_processors=data_processors,
            model=model,
            best_score=trainer.callback_metrics[f"val_{self._validation_metric_name}"].item(),
            strategy=strategy,
        )

    def predict_per_run(
        self,
        data: Union[pd.DataFrame, dict, list],
        realtime: Optional[bool],
        requires_label: bool,
        barebones: Optional[bool] = False,
    ) -> List[Dict]:
        """
        Perform inference for learner.

        Parameters
        ----------
        data
            The data for inference.
        realtime
            Whether use realtime inference.
        requires_label
            Whether uses label during inference.
        barebones
            Whether to run in “barebones mode”, where all lightning's features that may impact raw speed are disabled.

        Returns
        -------
        A list of output dicts.
        """
        data = self.on_predict_per_run_start(data=data)
        column_types = self.infer_column_types(
            column_types=self._column_types,
            data=data,
            is_train=False,
        )
        column_types = infer_rois_column_type(
            column_types=column_types,
            data=data,
        )
        df_preprocessor = self.get_df_preprocessor_per_run(
            df_preprocessor=self._df_preprocessor,
            data=data,
            column_types=column_types,
            is_train=False,
        )
        if self._fit_called:
            df_preprocessor._column_types = self.update_image_column_types(data=data)
        data_processors = self.get_data_processors_per_run(
            data_processors=self._data_processors,
            requires_label=requires_label,
            is_train=False,
        )
        num_gpus, strategy = self.get_num_gpus_and_strategy_per_run(
            predict_data=data,
            is_train=False,
        )
        precision = self.get_precision_per_run(
            num_gpus=num_gpus,
            precision=self._config.env.precision,
            cpu_only_warning=False,
        )
        batch_size = self.get_predict_batch_size_per_run(num_gpus=num_gpus, strategy=strategy)
        realtime, num_gpus, barebones = self.update_realtime_for_interactive_env(
            realtime=realtime,
            num_gpus=num_gpus,
            barebones=barebones,
            strategy=strategy,
        )
        datamodule = self.get_datamodule_per_run(
            df_preprocessor=df_preprocessor,
            data_processors=data_processors,
            per_gpu_batch_size=batch_size,
            num_workers=self._config.env.num_workers_evaluation,
            predict_data=data,
            is_train=False,
        )
        pred_writer = self.get_pred_writer(strategy=strategy)
        callbacks = self.get_callbacks_per_run(pred_writer=pred_writer, is_train=False)
        litmodule = self.get_litmodule_per_run(is_train=False)
        trainer = self.init_trainer_per_run(
            num_gpus=num_gpus,
            precision=precision,
            strategy=strategy,
            callbacks=callbacks,
            barebones=barebones,
            is_train=False,
        )
        outputs = self.run_trainer(
            trainer=trainer,
            litmodule=litmodule,
            datamodule=datamodule,
            pred_writer=pred_writer,
            is_train=False,
        )
        outputs = self.collect_predictions(
            outputs=outputs,
            trainer=trainer,
            pred_writer=pred_writer,
            num_gpus=num_gpus,
        )
        self.on_predict_per_run_end(trainer=trainer)

        # TODO: remove this by adjusting the return format of mmdet_image or lit_mmdet.
        if pred_writer is None and self._problem_type == OBJECT_DETECTION:
            outputs = [output for batch_outputs in outputs for output in batch_outputs]

        return outputs

    def evaluate_coco(
        self,
        anno_file_or_df: str,
        metrics: str,
        return_pred: Optional[bool] = False,
        eval_tool: Optional[str] = None,
    ):
        """
        Evaluate object detection model on a test dataset in COCO format.

        Parameters
        ----------
        anno_file_or_df
            The annotation file in COCO format
        metrics
            Metrics used for evaluation.
        return_pred
            Whether to return the prediction result of each row.
        eval_tool
            The eval_tool for object detection. Could be "pycocotools" or "torchmetrics".
        """
        if isinstance(anno_file_or_df, str):
            anno_file = anno_file_or_df
            data = from_coco_or_voc(
                anno_file, "test"
            )  # TODO: maybe remove default splits hardcoding (only used in VOC)
            if os.path.isdir(anno_file):
                eval_tool = "torchmetrics"  # we can only use torchmetrics for VOC format evaluation.
        else:
            # during validation, it will call evaluate with df as input
            anno_file = self._detection_anno_train
            data = anno_file_or_df

        outputs = self.predict_per_run(
            data=data,
            realtime=False,
            requires_label=True,
        )  # outputs shape: num_batch, 1(["bbox"]), batch_size, 2(if using mask_rcnn)/na, 80, n, 5

        # Cache prediction results as COCO format # TODO: refactor this
        self._save_path = setup_save_path(
            old_save_path=self._save_path,
            warn_if_exist=False,
        )
        cocoeval_cache_path = os.path.join(self._save_path, "object_detection_result_cache.json")
        eval_results = cocoeval(
            outputs=outputs,
            data=data,
            anno_file=anno_file,
            cache_path=cocoeval_cache_path,
            metrics=metrics,
            tool=eval_tool,
        )
        if return_pred:
            return eval_results, outputs
        else:
            return eval_results

    def evaluate(
        self,
        data: Union[pd.DataFrame, dict, list, str],
        metrics: Optional[Union[str, List[str]]] = None,
        return_pred: Optional[bool] = False,
        realtime: Optional[bool] = False,
        eval_tool: Optional[str] = None,
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
            Whether to do realtime inference, which is efficient for small data (default False).
            If provided None, we would infer it on based on the data modalities
            and sample number.
        eval_tool
            The eval_tool for object detection. Could be "pycocotools" or "torchmetrics".

        Returns
        -------
        A dictionary with the metric names and their corresponding scores.
        Optionally return a dataframe of prediction results.
        """
        self.ensure_predict_ready()
        if self._problem_type == OPEN_VOCABULARY_OBJECT_DETECTION:
            raise NotImplementedError("Open vocabulary object detection doesn't support calling `evaluate` yet.")

        if realtime:
            return NotImplementedError(f"Current problem type {self._problem_type} does not support realtime predict.")
        if isinstance(data, str):
            return self.evaluate_coco(
                anno_file_or_df=data,
                metrics=metrics,
                return_pred=return_pred,
                eval_tool=eval_tool,
            )
        else:
            data = object_detection_data_to_df(data)
            return self.evaluate_coco(
                anno_file_or_df=data,
                metrics=metrics,
                return_pred=return_pred,
                eval_tool="torchmetrics",
            )

    def predict(
        self,
        data: Union[pd.DataFrame, dict, list, str],
        as_pandas: Optional[bool] = None,
        realtime: Optional[bool] = False,
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
            Whether to do realtime inference, which is efficient for small data (default False).
            If provided None, we would infer it on based on the data modalities
            and sample number.
        save_results
            Whether to save the prediction results (only works for detection now)

        Returns
        -------
        Array of predictions, one corresponding to each row in given dataset.
        """
        self.ensure_predict_ready()
        ret_type = BBOX
        if self._problem_type == OBJECT_DETECTION:
            data = object_detection_data_to_df(data)
            if self._label_column not in data:
                self._label_column = None
        elif self._problem_type == OPEN_VOCABULARY_OBJECT_DETECTION:
            ret_type = OVD_RET

        outputs = self.predict_per_run(
            data=data,
            realtime=realtime,
            requires_label=False,
        )
        pred = extract_from_output(outputs=outputs, ret_type=ret_type)
        if self._problem_type == OBJECT_DETECTION:
            if self._model.output_bbox_format == XYWH:
                pred = convert_pred_to_xywh(pred)

        if save_results and self._problem_type == OBJECT_DETECTION:
            self._save_path = setup_save_path(
                old_save_path=self._save_path,
                warn_if_exist=False,
            )
            save_result_df(
                pred=pred,
                data=data,
                detection_classes=self._model.model.CLASSES,
                result_path=os.path.join(self._save_path, "result.txt"),
            )

        if (as_pandas is None and isinstance(data, pd.DataFrame)) or as_pandas is True:
            if (
                self._problem_type == OBJECT_DETECTION
            ):  # TODO: add prediction output in COCO format if as_pandas is False
                # TODO: calling save_result_df to convert data to dataframe is not a good logic
                # TODO: consider combining this with the above saving logic or using a different function.
                pred = save_result_df(
                    pred=pred,
                    data=data,
                    detection_classes=self._model.model.CLASSES,
                    result_path=None,
                )
            elif (
                self._problem_type == OPEN_VOCABULARY_OBJECT_DETECTION
            ):  # TODO: refactor and merge with OBJECT DETECTION
                pred = save_ovd_result_df(
                    pred=pred,
                    data=data,
                    result_path=None,
                )

        return pred

    def predict_proba(
        self,
        data: Union[pd.DataFrame, dict, list],
        as_pandas: Optional[bool] = None,
        as_multiclass: Optional[bool] = True,
        realtime: Optional[bool] = False,
        **kwargs,
    ):
        raise NotImplementedError("Object detection doesn't support calling `predict_proba` yet.")

    def extract_embedding(
        self,
        data: Union[pd.DataFrame, dict, list],
        as_tensor: Optional[bool] = False,
        as_pandas: Optional[bool] = False,
        realtime: Optional[bool] = False,
        **kwargs,
    ):
        raise NotImplementedError("Object detection doesn't support calling `extract_embedding` yet.")

    @staticmethod
    def _load_metadata(
        learner,
        path: str,
        resume: Optional[bool] = False,
        verbosity: Optional[int] = 3,
    ):
        learner = super()._load_metadata(learner=learner, path=path, resume=resume, verbosity=verbosity)
        learner._data_processors = None
        return learner

    def save(
        self,
        path: str,
        standalone: Optional[bool] = True,
        config: Optional[DictConfig] = None,
        model: Optional[nn.Module] = None,
        df_preprocessor: Optional[MultiModalFeaturePreprocessor] = None,
        data_processors: Optional[Dict] = None,
        fit_called: Optional[bool] = None,
        save_model: Optional[bool] = True,
    ):
        super().save(
            path=path,
            standalone=standalone,
            config=config,
            model=model,
            df_preprocessor=df_preprocessor,
            data_processors=data_processors,
            fit_called=fit_called,
            save_model=save_model,
        )
        assets_path = os.path.join(path, "assets.json")
        with open(assets_path, "r") as fp:
            assets = json.load(fp)
            assets.update(
                {
                    "classes": self._classes,
                }
            )
        os.remove(assets_path)
        with open(assets_path, "w") as fp:
            json.dump(assets, fp, ensure_ascii=True)
