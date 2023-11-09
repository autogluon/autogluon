import logging
import sys
import warnings
from datetime import timedelta
from typing import Dict, List, Optional, Union

import pandas as pd
from omegaconf import OmegaConf
from torch import nn

from autogluon.core.metrics import Scorer

from ..constants import NER, NER_RET, OVERALL_F1, Y_PRED, Y_TRUE
from ..data import MultiModalFeaturePreprocessor
from ..optimization import NerLitModule
from ..utils import (
    compute_inference_batch_size,
    compute_num_gpus,
    compute_score,
    extract_from_output,
    infer_precision,
    merge_bio_format,
)
from .base import BaseLearner

logger = logging.getLogger(__name__)


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
        warn_if_exist: Optional[bool] = True,
        enable_progress_bar: Optional[bool] = None,
        pretrained: Optional[bool] = True,
        validation_metric: Optional[str] = None,
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
            warn_if_exist=warn_if_exist,
            enable_progress_bar=enable_progress_bar,
            pretrained=pretrained,
            validation_metric=validation_metric,
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

    def get_optimization_kwargs_per_run(self, config, validation_metric, custom_metric_func, loss_func):
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
            loss_func=loss_func,
            efficient_finetune=OmegaConf.select(config, "optimization.efficient_finetune"),
            skip_final_val=OmegaConf.select(config, "optimization.skip_final_val", default=False),
        )

    def get_litmodule_per_run(
        self,
        model: Optional[nn.Module] = None,
        peft_param_names: Optional[List[str]] = None,
        optimization_kwargs: Optional[dict] = None,
        is_train=True,
    ):
        if is_train:
            return NerLitModule(
                model=model,
                trainable_param_names=peft_param_names,
                **optimization_kwargs,
            )
        else:
            return NerLitModule(model=self._model)

    @staticmethod
    def get_output_shape_per_run(df_preprocessor):
        # ner needs to update output_shape with label_generator.
        return len(df_preprocessor.label_generator.unique_entity_groups)

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
        self._output_shape = self.get_output_shape_per_run(df_preprocessor=df_preprocessor)
        model = self.get_model_per_run(model=model, config=config, df_preprocessor=df_preprocessor)
        model = self.compile_model_per_run(config=config, model=model)
        peft_param_names = self.get_peft_param_names_per_run(model=model, config=config)
        data_processors = self.get_data_processors_per_run(
            data_processors=data_processors,
            config=config,
            model=model,
            advanced_hyperparameters=advanced_hyperparameters,
        )
        validation_metric, custom_metric_func = self.get_validation_metric_per_run()
        loss_func = self.get_loss_func_per_run(config=config)
        if max_time == timedelta(seconds=0):
            return dict(
                config=config,
                df_preprocessor=df_preprocessor,
                data_processors=data_processors,
                model=model,
                strict_loading=not peft_param_names,
            )

        datamodule = self.get_datamodule_per_run(
            df_preprocessor=df_preprocessor,
            data_processors=data_processors,
            per_gpu_batch_size=config.env.per_gpu_batch_size,
            num_workers=config.env.num_workers,
        )
        optimization_kwargs = self.get_optimization_kwargs_per_run(
            config=config,
            validation_metric=validation_metric,
            custom_metric_func=custom_metric_func,
            loss_func=loss_func,
        )
        litmodule = self.get_litmodule_per_run(
            model=model,
            peft_param_names=peft_param_names,
            optimization_kwargs=optimization_kwargs,
        )
        callbacks = self.get_callbacks_per_run(save_path=save_path, config=config, litmodule=litmodule)
        plugins = self.get_plugins_per_run(model=model, peft_param_names=peft_param_names)
        tb_logger = self.get_tb_logger(save_path=save_path)
        num_gpus = compute_num_gpus(config_num_gpus=config.env.num_gpus, strategy=config.env.strategy)
        self.log_gpu_info(num_gpus=num_gpus, config=config)
        precision = infer_precision(num_gpus=num_gpus, precision=config.env.precision)
        grad_steps = self.get_grad_steps(num_gpus=num_gpus, config=config)
        strategy = self.get_strategy_per_run(num_gpus=num_gpus, config=config)
        strategy, num_gpus = self.update_strategy_and_num_gpus_for_hpo(strategy=strategy, num_gpus=num_gpus)
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
            strict_loading=not peft_param_names,
        )

    def evaluate(
        self,
        data: Union[pd.DataFrame, dict, list, str],
        metrics: Optional[Union[str, List[str]]] = None,
        return_pred: Optional[bool] = False,
        realtime: Optional[bool] = None,
        **kwargs,
    ):
        """ """
        self.ensure_predict_ready()
        outputs = self.predict_per_run(
            data=data,
            requires_label=True,
            realtime=realtime,
        )
        logits = extract_from_output(ret_type=NER_RET, outputs=outputs)
        metric_data = {}
        y_pred = self._df_preprocessor.transform_prediction(
            y_pred=logits,
            inverse_categorical=False,
        )
        y_pred_inv = self._df_preprocessor.transform_prediction(
            y_pred=logits,
            inverse_categorical=True,
        )
        y_true = self._df_preprocessor.transform_label_for_metric(
            df=data,
            tokenizer=self._model.tokenizer,
        )
        metric_data.update(
            {
                Y_PRED: y_pred,
                Y_TRUE: y_true,
            }
        )
        metrics_is_none = False
        if metrics is None:
            metrics_is_none = True
            if self._eval_metric_func:
                metrics = [self._eval_metric_func]
            else:
                metrics = [self._eval_metric_name]
        if isinstance(metrics, str) or isinstance(metrics, Scorer):
            metrics = [metrics]

        results = {}
        score = compute_score(
            metric_data=metric_data,
            metric=self._eval_metric_name.lower(),
        )
        score = {k.lower(): v for k, v in score.items()}
        if metrics_is_none:
            results = score
        else:
            for per_metric in metrics:
                if per_metric.lower() in score:
                    results.update({per_metric: score[per_metric.lower()]})
                else:
                    warnings.warn(f"Warning: {per_metric} is not a supported evaluation metric!")
            if not results:
                results = score  # If the results dict is empty, return all scores.

        if return_pred:
            return results, self._as_pandas(data=data, to_be_converted=y_pred_inv)
        else:
            return results

    def predict(
        self,
        data: Union[pd.DataFrame, dict, list, str],
        as_pandas: Optional[bool] = None,
        realtime: Optional[bool] = None,
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

        Returns
        -------
        Array of predictions, one corresponding to each row in given dataset.
        """
        self.ensure_predict_ready()
        outputs = self.predict_per_run(
            data=data,
            requires_label=False,
            realtime=realtime,
        )
        logits = extract_from_output(outputs=outputs, ret_type=NER_RET)
        if self._df_preprocessor:
            pred = self._df_preprocessor.transform_prediction(
                y_pred=logits,
            )
        else:
            pred = logits

        pred = merge_bio_format(data[self._df_preprocessor.ner_feature_names[0]], pred)
        if (as_pandas is None and isinstance(data, pd.DataFrame)) or as_pandas is True:
            pred = self._as_pandas(data=data, to_be_converted=pred)

        return pred

    def predict_proba(
        self,
        data: Union[pd.DataFrame, dict, list],
        as_pandas: Optional[bool] = None,
        realtime: Optional[bool] = None,
        **kwargs,
    ):
        """
        Predict probabilities class probabilities rather than class labels.
        This is only for the classification. Calling it for a regression will throw an exception.

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

        Returns
        -------
        Array of predicted class-probabilities, corresponding to each row in the given data.
        When as_multiclass is True, the output will always have shape (#samples, #classes).
        Otherwise, the output will have shape (#samples,)
        """
        self.ensure_predict_ready()
        outputs = self.predict_per_run(
            data=data,
            requires_label=False,
            realtime=realtime,
        )
        ner_outputs = extract_from_output(outputs=outputs, ret_type=NER_RET)
        prob = self._df_preprocessor.transform_prediction(
            y_pred=ner_outputs,
            return_proba=True,
        )
        if (as_pandas is None and isinstance(data, pd.DataFrame)) or as_pandas is True:
            prob = self._as_pandas(data=data, to_be_converted=prob)

        return prob
