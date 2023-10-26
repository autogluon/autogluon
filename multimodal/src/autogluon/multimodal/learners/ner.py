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

    def evaluate(
        self,
        data: Union[pd.DataFrame, dict, list, str],
        metrics: Optional[Union[str, List[str]]] = None,
        return_pred: Optional[bool] = False,
        realtime: Optional[bool] = None,
        eval_tool: Optional[str] = None,
    ):
        """
        """
        self._ensure_inference_ready()
        ret_type = NER_RET
        outputs = predict(
            predictor=self,
            data=data,
            requires_label=True,
            realtime=realtime,
        )
        logits = extract_from_output(ret_type=ret_type, outputs=outputs)
        metric_data = {}
        y_pred = self._df_preprocessor.transform_prediction(
            y_pred=logits,
            inverse_categorical=False,
        )
        y_pred_inv = self._df_preprocessor.transform_prediction(
            y_pred=logits,
            inverse_categorical=True,
        )
        y_true = self._df_preprocessor.transform_label_for_metric(df=data, tokenizer=self._model.tokenizer)
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
                    logger.warning(f"Warning: {per_metric} is not a supported evaluation metric!")
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
            save_results: Optional[bool] = None,
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
        self._ensure_inference_ready()
        ret_type = NER_RET
        outputs = predict(
            predictor=self,
            data=data,
            requires_label=False,
            realtime=realtime,
        )
        logits = extract_from_output(outputs=outputs, ret_type=ret_type)
        if self._df_preprocessor:
            pred = self._df_preprocessor.transform_prediction(
                y_pred=logits,
            )
        else:
            if isinstance(logits, (torch.Tensor, np.ndarray)) and logits.ndim == 2:
                pred = logits.argmax(axis=1)
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
    ):
        """
        Predict probabilities class probabilities rather than class labels.
        This is only for the classification tasks. Calling it for a regression task will throw an exception.

        Parameters
        ----------
        data
            The data to make predictions for. Should contain same column names as training data and
              follow same format (except for the `label` column).
        candidate_data
            The candidate data from which to search the query data's matches.
        id_mappings
             Id-to-content mappings. The contents can be text, image, etc.
             This is used when data contain the query/response identifiers instead of their contents.
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
        When as_multiclass is True, the output will always have shape (#samples, #classes).
        Otherwise, the output will have shape (#samples,)
        """
        self._ensure_inference_ready()

        outputs = predict(
            predictor=self,
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

    def extract_embedding(
            self,
    ):
        raise RuntimeError("NER doesn't support calling `extract_embedding`.")