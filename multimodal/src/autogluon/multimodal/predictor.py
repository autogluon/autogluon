"""Implementation of the multimodal predictor"""

from __future__ import annotations

import json
import logging
import os
import warnings
from typing import Dict, List, Optional, Union

import pandas as pd
import transformers

from autogluon.common.utils.log_utils import set_logger_verbosity, verbosity2loglevel
from autogluon.core.metrics import Scorer

from .constants import AUTOMM_TUTORIAL_MODE, FEW_SHOT_CLASSIFICATION, NER, OBJECT_DETECTION, SEMANTIC_SEGMENTATION
from .learners import (
    BaseLearner,
    FewShotSVMLearner,
    MultiModalMatcher,
    NERLearner,
    ObjectDetectionLearner,
    SemanticSegmentationLearner,
)
from .problem_types import PROBLEM_TYPES_REG
from .utils import get_dir_ckpt_paths

pl_logger = logging.getLogger("lightning")
pl_logger.propagate = False  # https://github.com/Lightning-AI/lightning/issues/4621
logger = logging.getLogger(__name__)


class MultiModalPredictor:
    """
    AutoMM is designed to simplify the fine-tuning of foundation models
    for downstream applications with just three lines of code.
    AutoMM seamlessly integrates with popular model zoos such as
    `HuggingFace Transformers <https://github.com/huggingface/transformers>`_,
    `TIMM <https://github.com/huggingface/pytorch-image-models>`_,
    and `MMDetection <https://github.com/open-mmlab/mmdetection>`_,
    accommodating a diverse range of data modalities,
    including image, text, tabular, and document data, whether used individually or in combination.
    It offers support for an array of tasks, encompassing classification, regression,
    object detection, named entity recognition, semantic matching, and image segmentation.
    """

    def __init__(
        self,
        label: Optional[str] = None,
        problem_type: Optional[str] = None,
        query: Optional[Union[str, List[str]]] = None,
        response: Optional[Union[str, List[str]]] = None,
        match_label: Optional[Union[int, str]] = None,
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
    ):
        """
        Parameters
        ----------
        label
            Name of one pd.DataFrame column that contains the target variable to predict.
        problem_type
            Type of problem. We support standard problems like

            - 'binary': Binary classification
            - 'multiclass': Multi-class classification
            - 'regression': Regression
            - 'classification': Classification problems include 'binary' and 'multiclass' classification.

            In addition, we support advanced problems such as

            - 'object_detection': Object detection
            - 'ner' or 'named_entity_recognition': Named entity extraction
            - 'text_similarity': Text-text semantic matching
            - 'image_similarity': Image-image semantic matching
            - 'image_text_similarity': Text-image semantic matching
            - 'feature_extraction': Extracting feature (only support inference)
            - 'zero_shot_image_classification': Zero-shot image classification (only support inference)
            - 'few_shot_classification': Few-shot classification for image or text data.
            - 'semantic_segmentation': Semantic segmentation with Segment Anything Model.

            For certain problem types, the default behavior is to load a pretrained model based on
            the presets / hyperparameters and the predictor can do zero-shot inference
            (running inference without .fit()). Those include the following
            problem types:

            - 'object_detection'
            - 'text_similarity'
            - 'image_similarity'
            - 'image_text_similarity'
            - 'feature_extraction'
            - 'zero_shot_image_classification'

        query
            Name of one pd.DataFrame column that has the query data in semantic matching tasks.
        response
            Name of one pd.DataFrame column that contains the response data in semantic matching tasks.
            If no label column is provided, the query and response pairs in
            one pd.DataFrame row are assumed to be positive pairs.
        match_label
            The label class that indicates the <query, response> pair is counted as a "match".
            This is used when the task belongs to semantic matching, and the labels are binary.
            For example, the label column can contain ["duplicate", "not duplicate"] in a duplicate detection task.
            The match_label should be "duplicate" since it means that two items match.
        presets
            Presets regarding model quality, e.g., 'best_quality', 'high_quality' (default), and 'medium_quality'.
            Each quality has its corresponding HPO presets: 'best_quality_hpo', 'high_quality_hpo', and 'medium_quality_hpo'.
        eval_metric
            Evaluation metric name. If `eval_metric = None`, it is automatically chosen based on `problem_type`.
            Defaults to 'accuracy' for multiclass classification, `roc_auc` for binary classification,
            and 'root_mean_squared_error' for regression.
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
            Path to directory where models and related artifacts should be saved.
            If unspecified, a time-stamped folder called "AutogluonAutoMM/ag-[TIMESTAMP]"
            will be created in the working directory.
            Note: To call `fit()` twice and save all results of each fit,
            you must specify different `path` locations or don't specify `path` at all.
        verbosity
            Verbosity levels range from 0 to 4, controlling how much logging information is printed.
            Higher levels correspond to more detailed print statements.
            You can set verbosity = 0 to suppress warnings.
        num_classes
            Number of classes (used for object detection).
            If this is specified and is different from the pretrained model's output shape,
            the model's head will be changed to have <num_classes> output.
        classes
            All the classes (used for object detection).
        warn_if_exist
            Whether to raise warning if the specified path already exists (Default True).
        enable_progress_bar
            Whether to show progress bar (default True). It would be
            disabled if the environment variable os.environ["AUTOMM_DISABLE_PROGRESS_BAR"] is set.
        pretrained
            Whether to initialize the model with pretrained weights (default True).
            If False, it creates a model with random initialization.
        validation_metric
            Validation metric for selecting the best model and early-stopping during training.
            If not provided, it would be automatically chosen based on the problem type.
        sample_data_path
            The path to sample data from which we can infer num_classes or classes used for object detection.
        """
        if problem_type is not None:
            problem_type = problem_type.lower()
            assert problem_type in PROBLEM_TYPES_REG, (
                f"problem_type='{problem_type}' is not supported yet. You may pick a problem type from"
                f" {PROBLEM_TYPES_REG.list_keys()}."
            )
            problem_property = PROBLEM_TYPES_REG.get(problem_type)
            if problem_property.experimental:
                warnings.warn(
                    f"problem_type='{problem_type}' is currently experimental.",
                    UserWarning,
                )
            problem_type = problem_property.name
        else:
            problem_property = None

        if os.environ.get(AUTOMM_TUTORIAL_MODE):
            enable_progress_bar = False
            # Also disable progress bar of transformers package
            transformers.logging.disable_progress_bar()

        if verbosity is not None:
            set_logger_verbosity(verbosity)

        self._verbosity = verbosity

        if problem_property and problem_property.is_matching:
            learner_class = MultiModalMatcher
        elif problem_type == OBJECT_DETECTION:
            learner_class = ObjectDetectionLearner
        elif problem_type == NER:
            learner_class = NERLearner
        elif problem_type == FEW_SHOT_CLASSIFICATION:
            learner_class = FewShotSVMLearner
        elif problem_type == SEMANTIC_SEGMENTATION:
            learner_class = SemanticSegmentationLearner
        else:
            learner_class = BaseLearner

        self._learner = learner_class(
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
            sample_data_path=sample_data_path,
            validation_metric=validation_metric,
            query=query,
            response=response,
            match_label=match_label,
        )

    @property
    def path(self):
        """
        Path to directory where the model and related artifacts are stored.
        """
        return self._learner.path

    @property
    def label(self):
        """
        Name of one pd.DataFrame column that contains the target variable to predict.
        """
        return self._learner.label

    @property
    def query(self):
        """
        Name of one pd.DataFrame column that has the query data in semantic matching tasks.
        """
        return self._learner.query

    @property
    def response(self):
        """
        Name of one pd.DataFrame column that contains the response data in semantic matching tasks.
        """
        return self._learner.response

    @property
    def match_label(self):
        """
        The label class that indicates the <query, response> pair is counted as "match" in the semantic matching tasks.
        """
        return self._learner.match_label

    @property
    def problem_type(self):
        """
        What type of prediction problem this predictor has been trained for.
        """
        return self._learner.problem_type

    @property
    def problem_property(self):
        """
        Property of the problem, storing the problem type and its related properties.
        """
        return self._learner.problem_property

    @property
    def column_types(self):
        """
        Column types in the pd.DataFrame.
        """
        return self._learner.column_types

    @property
    def eval_metric(self):
        """
        What metric is used to evaluate predictive performance.
        """
        return self._learner.eval_metric

    @property
    def validation_metric(self):
        """
        Validation metric for selecting the best model and early-stopping during training.
        Note that the validation metric may be different from the evaluation metric.
        """
        return self._learner.validation_metric

    @property
    def verbosity(self):
        """
        Verbosity levels range from 0 to 4 and control how much information is printed.
        Higher levels correspond to more detailed print statements.
        """
        return self._verbosity

    @property
    def total_parameters(self) -> int:
        """
        The number of model parameters.
        """
        return self._learner.total_parameters

    @property
    def trainable_parameters(self) -> int:
        """
        The number of trainable model parameters, usually those with requires_grad=True.
        """
        return self._learner.trainable_parameters

    @property
    def model_size(self) -> float:
        """
        Returns the model size in Megabyte.
        """
        return self._learner.model_size

    @property
    def classes(self):
        """
        Object classes for the object detection problem type.
        """
        return self._learner.classes

    @property
    def class_labels(self):
        """
        The original name of the class labels.
        For example, the tabular data may contain classes equal to
        "entailment", "contradiction", "neutral". Internally, these will be converted to
        0, 1, 2, ...
        This function returns the original names of these raw labels.

        Returns
        -------
        List that contain the class names. It will be None if it's not a classification problem.
        """
        return self._learner.class_labels

    @property
    def positive_class(self):
        """
        Name of the class label that will be mapped to 1.
        This is only meaningful for binary classification problems.

        It is useful for computing metrics such as F1 which require a positive and negative class.
        You may refer to https://en.wikipedia.org/wiki/F-score for more details.
        In binary classification, :class:`MultiModalPredictor.predict_proba(as_multiclass=False)`
        returns the estimated probability that each row belongs to the positive class.
        Will print a warning and return None if called when `predictor.problem_type != 'binary'`.

        Returns
        -------
        The positive class name in binary classification or None if the problem is not binary classification.
        """
        return self._learner.positive_class

    # This func is required by the abstract trainer of TabularPredictor.
    def set_verbosity(self, verbosity: int):
        """Set the verbosity level of the log.

        Parameters
        ----------
        verbosity
            The verbosity level.

            0 --> only errors
            1 --> only warnings and critical print statements
            2 --> key print statements which should be shown by default
            3 --> more-detailed printing
            4 --> everything

        """
        self._verbosity = verbosity
        set_logger_verbosity(verbosity)
        # TODO: align verbosity2loglevel with https://huggingface.co/docs/transformers/main_classes/logging#transformers.utils.logging.get_verbosity

    def set_num_gpus(self, num_gpus):
        """
        Set the number of GPUs in config.
        """
        self._learner.set_num_gpus(num_gpus)

    def get_num_gpus(self):
        """
        Get the number of GPUs from config.
        """
        self._learner.get_num_gpus()

    def fit(
        self,
        train_data: Union[pd.DataFrame, str],
        presets: Optional[str] = None,
        tuning_data: Optional[Union[pd.DataFrame, str]] = None,
        max_num_tuning_data: Optional[int] = None,
        id_mappings: Optional[Union[Dict[str, Dict], Dict[str, pd.Series]]] = None,
        time_limit: Optional[int] = None,
        save_path: Optional[str] = None,
        hyperparameters: Optional[Union[str, Dict, List[str]]] = None,
        column_types: Optional[dict] = None,
        holdout_frac: Optional[float] = None,
        teacher_predictor: Union[str, MultiModalPredictor] = None,
        seed: Optional[int] = 0,
        standalone: Optional[bool] = True,
        hyperparameter_tune_kwargs: Optional[dict] = None,
        clean_ckpts: Optional[bool] = True,
    ):
        """
        Fit models to predict a column of a data table (label) based on the other columns (features).

        Parameters
        ----------
        train_data
            A pd.DataFrame containing training data.
        presets
            Presets regarding model quality, e.g., best_quality, high_quality, and medium_quality.
            Each quality has its corresponding HPO presets: 'best_quality_hpo', 'high_quality_hpo', and 'medium_quality_hpo'.
        tuning_data
            A pd.DataFrame containing validation data, which should have the same columns as the train_data.
            If `tuning_data = None`, `fit()` will automatically hold out some random validation data from `train_data`.
        max_num_tuning_data
            The maximum number of tuning samples (used for object detection).
        id_mappings
             Id-to-content mappings (used for semantic matching). The contents can be text, image, etc.
             This is used when the pd.DataFrame contains the query/response identifiers instead of their contents.
        time_limit
            How long `fit()` should run for (wall clock time in seconds).
            If not specified, `fit()` will run until the model has completed training.
        save_path
            Path to directory where models and artifacts should be saved.
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
        column_types
            A dictionary that maps column names to their data types.
            For example: `column_types = {"item_name": "text", "image": "image_path",
            "product_description": "text", "height": "numerical"}`
            may be used for a table with columns: "item_name", "brand", "product_description", and "height".
            If None, column_types will be automatically inferred from the data.
            The current supported types are:
                - "image_path": each row in this column is one image path.
                - "text": each row in this column contains text (sentence, paragraph, etc.).
                - "numerical": each row in this column contains a number.
                - "categorical": each row in this column belongs to one of K categories.
        holdout_frac
            Fraction of train_data to holdout as tuning_data for optimizing hyperparameters or
            early stopping (ignored unless `tuning_data = None`).
            Default value (if None) is selected based on the number of rows in the training data
            and whether hyperparameter optimization is utilized.
        teacher_predictor
            The pre-trained teacher predictor or its saved path. If provided, `fit()` can distill its
            knowledge to a student predictor, i.e., the current predictor.
        seed
            The random seed to be used for training (default 0).
        standalone
            Whether to save the entire model for offline deployment.
        hyperparameter_tune_kwargs
                Hyperparameter tuning strategy and kwargs (for example, how many HPO trials to run).
                If None, then hyperparameter tuning will not be performed.
                    num_trials: int
                        How many HPO trials to run. Either `num_trials` or `time_limit` to `fit` needs to be specified.
                    scheduler: Union[str, ray.tune.schedulers.TrialScheduler]
                        If str is passed, AutoGluon will create the scheduler for you with some default parameters.
                        If ray.tune.schedulers.TrialScheduler object is passed, you are responsible for initializing the object.
                    scheduler_init_args: Optional[dict] = None
                        If provided str to `scheduler`, you can optionally provide custom init_args to the scheduler
                    searcher: Union[str, ray.tune.search.SearchAlgorithm, ray.tune.search.Searcher]
                        If str is passed, AutoGluon will create the searcher for you with some default parameters.
                        If ray.tune.schedulers.TrialScheduler object is passed, you are responsible for initializing the object.
                        You don't need to worry about `metric` and `mode` of the searcher object. AutoGluon will figure it out by itself.
                    scheduler_init_args: Optional[dict] = None
                        If provided str to `searcher`, you can optionally provide custom init_args to the searcher
                        You don't need to worry about `metric` and `mode`. AutoGluon will figure it out by itself.
        clean_ckpts
            Whether to clean the intermediate checkpoints after training.

        Returns
        -------
        An "MultiModalPredictor" object (itself).
        """

        if teacher_predictor is None:
            teacher_learner = None
        elif isinstance(teacher_predictor, str):
            teacher_learner = teacher_predictor
        else:
            teacher_learner = teacher_predictor._learner
        self._learner.fit(
            train_data=train_data,
            presets=presets,
            tuning_data=tuning_data,
            max_num_tuning_data=max_num_tuning_data,
            time_limit=time_limit,
            save_path=save_path,
            hyperparameters=hyperparameters,
            column_types=column_types,
            holdout_frac=holdout_frac,
            teacher_learner=teacher_learner,
            seed=seed,
            standalone=standalone,
            hyperparameter_tune_kwargs=hyperparameter_tune_kwargs,
            clean_ckpts=clean_ckpts,
            id_mappings=id_mappings,
        )

        return self

    def evaluate(
        self,
        data: Union[pd.DataFrame, dict, list, str],
        query_data: Optional[list] = None,
        response_data: Optional[list] = None,
        id_mappings: Optional[Union[Dict[str, Dict], Dict[str, pd.Series]]] = None,
        metrics: Optional[Union[str, List[str]]] = None,
        chunk_size: Optional[int] = 1024,
        similarity_type: Optional[str] = "cosine",
        cutoffs: Optional[List[int]] = [1, 5, 10],
        label: Optional[str] = None,
        return_pred: Optional[bool] = False,
        realtime: Optional[bool] = False,
        eval_tool: Optional[str] = None,
    ):
        """
        Evaluate the model on a given dataset.

        Parameters
        ----------
        data
            A pd.DataFrame, containing the same columns as the training data.
            Or a str, that is a path of the annotation file for detection.
        query_data
            Query data used for ranking.
        response_data
            Response data used for ranking.
        id_mappings
             Id-to-content mappings. The contents can be text, image, etc.
             This is used when data/query_data/response_data contain the query/response identifiers instead of their contents.
        metrics
            A list of metric names to report.
            If None, we only return the score for the stored `_eval_metric_name`.
        chunk_size
            Scan the response data by chunk_size each time. Increasing the value increases the speed, but requires more memory.
        similarity_type
            Use what function (cosine/dot_prod) to score the similarity (default: cosine).
        cutoffs
            A list of cutoff values to evaluate ranking.
        label
            The label column name in data. Some tasks, e.g., image<-->text matching, have no label column in training data,
            but the label column may be still required in evaluation.
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
        Optionally return a pd.DataFrame of prediction results.
        """
        return self._learner.evaluate(
            data=data,
            metrics=metrics,
            return_pred=return_pred,
            realtime=realtime,
            eval_tool=eval_tool,
            query_data=query_data,
            response_data=response_data,
            id_mappings=id_mappings,
            chunk_size=chunk_size,
            similarity_type=similarity_type,
            cutoffs=cutoffs,
            label=label,
        )

    def predict(
        self,
        data: Union[pd.DataFrame, dict, list, str],
        candidate_data: Optional[Union[pd.DataFrame, dict, list]] = None,
        id_mappings: Optional[Union[Dict[str, Dict], Dict[str, pd.Series]]] = None,
        as_pandas: Optional[bool] = None,
        realtime: Optional[bool] = False,
        save_results: Optional[bool] = None,
    ):
        """
        Predict the label column values for new data.

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
        return self._learner.predict(
            data=data,
            candidate_data=candidate_data,
            as_pandas=as_pandas,
            realtime=realtime,
            save_results=save_results,
            id_mappings=id_mappings,
        )

    def predict_proba(
        self,
        data: Union[pd.DataFrame, dict, list],
        candidate_data: Optional[Union[pd.DataFrame, dict, list]] = None,
        id_mappings: Optional[Union[Dict[str, Dict], Dict[str, pd.Series]]] = None,
        as_pandas: Optional[bool] = None,
        as_multiclass: Optional[bool] = True,
        realtime: Optional[bool] = False,
    ):
        """
        Predict class probabilities rather than class labels.
        Note that this is only for the classification tasks.
        Calling it for a regression task will throw an exception.

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
            Whether to do realtime inference, which is efficient for small data (default False).
            If provided None, we would infer it on based on the data modalities
            and sample number.

        Returns
        -------
        Array of predicted class-probabilities, corresponding to each row in the given data.
        When as_multiclass is True, the output will always have shape (#samples, #classes).
        Otherwise, the output will have shape (#samples,)
        """
        return self._learner.predict_proba(
            data=data,
            candidate_data=candidate_data,
            as_pandas=as_pandas,
            as_multiclass=as_multiclass,
            realtime=realtime,
            id_mappings=id_mappings,
        )

    def extract_embedding(
        self,
        data: Union[pd.DataFrame, dict, list],
        id_mappings: Optional[Union[Dict[str, Dict], Dict[str, pd.Series]]] = None,
        return_masks: Optional[bool] = False,
        as_tensor: Optional[bool] = False,
        as_pandas: Optional[bool] = False,
        realtime: Optional[bool] = False,
        signature: Optional[str] = None,
    ):
        """
        Extract features for each sample, i.e., one row in the provided pd.DataFrame `data`.

        Parameters
        ----------
        data
            The data to extract embeddings for. Should contain same column names as training dataset and
            follow same format (except for the `label` column).
        id_mappings
             Id-to-content mappings. The contents can be text, image, etc.
             This is used when data contain the query/response identifiers instead of their contents.
        return_masks
            If true, returns a mask dictionary, whose keys are the same as those in the features dictionary.
            If a sample has empty input in feature column `image_0`, the sample will has mask 0 under key `image_0`.
        as_tensor
            Whether to return a Pytorch tensor.
        as_pandas
            Whether to return the output as a pandas DataFrame (True) or numpy array (False).
        realtime
            Whether to do realtime inference, which is efficient for small data (default False).
            If provided None, we would infer it on based on the data modalities
            and sample number.
        signature
            When using matcher, it can be query or response.

        Returns
        -------
        Array of embeddings, corresponding to each row in the given data.
        It will have shape (#samples, D) where the embedding dimension D is determined
        by the neural network's architecture.
        """
        return self._learner.extract_embedding(
            data=data,
            return_masks=return_masks,
            as_tensor=as_tensor,
            as_pandas=as_pandas,
            realtime=realtime,
            signature=signature,
            id_mappings=id_mappings,
        )

    def save(self, path: str, standalone: Optional[bool] = True):
        """
        Save this predictor to file in directory specified by `path`.

        Parameters
        ----------
        path
            The directory to save this predictor.
        standalone
            Whether to save the downloaded model for offline deployment.
            When standalone = True, save the transformers.CLIPModel and transformers.AutoModel to os.path.join(path,model_name),
            and reset the associate model.model_name.checkpoint_name start with `local://` in config.yaml.
            When standalone = False, the saved artifact may require an online environment to process in load().
        """
        self._learner.save(path=path, standalone=standalone)

    @classmethod
    def load(
        cls,
        path: str,
        resume: Optional[bool] = False,
        verbosity: Optional[int] = 3,
    ):
        """
        Load a predictor object from a directory specified by `path`. The to-be-loaded predictor
        can be completely or partially trained by .fit(). If a previous training has completed,
        it will load the checkpoint `model.ckpt`. Otherwise, if a previous training accidentally
        collapses in the middle, it can load the `last.ckpt` checkpoint by setting `resume=True`.
        It also supports loading one specific checkpoint given its path.

        .. warning::

            :meth:`autogluon.multimodal.MultiModalPredictor.load` uses `pickle` module implicitly, which is known to
            be insecure. It is possible to construct malicious pickle data which will execute arbitrary code during
            unpickling. Never load data that could have come from an untrusted source, or that could have been tampered
            with. **Only load data you trust.**

        Parameters
        ----------
        path
            The directory to load the predictor object.
        resume
            Whether to resume training from `last.ckpt`. This is useful when a training was accidentally
            broken during the middle, and we want to resume the training from the last saved checkpoint.
        verbosity
            Verbosity levels range from 0 to 4 and control how much information is printed.
            Higher levels correspond to more detailed print statements.
            You can set verbosity = 0 to suppress warnings.

        Returns
        -------
        The loaded predictor object.
        """
        dir_path, ckpt_path = get_dir_ckpt_paths(path=path)

        assert os.path.isdir(dir_path), f"'{dir_path}' must be an existing directory."
        predictor = cls(label="dummy_label")

        with open(os.path.join(dir_path, "assets.json"), "r") as fp:
            assets = json.load(fp)
        if "class_name" in assets and assets["class_name"] == "MultiModalMatcher":
            learner_class = MultiModalMatcher
        elif assets["problem_type"] == OBJECT_DETECTION:
            learner_class = ObjectDetectionLearner
        elif assets["problem_type"] == NER:
            learner_class = NERLearner
        elif assets["problem_type"] == FEW_SHOT_CLASSIFICATION:
            learner_class = FewShotSVMLearner
        elif assets["problem_type"] == SEMANTIC_SEGMENTATION:
            learner_class = SemanticSegmentationLearner
        else:
            learner_class = BaseLearner

        predictor._learner = learner_class.load(path=path, resume=resume, verbosity=verbosity)
        return predictor

    def dump_model(self, save_path: Optional[str] = None):
        """
        Save model weights and config to a local directory.
        Model weights are saved in the file `pytorch_model.bin` (for `timm_image` or `hf_text`)
        or '<ckpt_name>.pth' (for `mmdet_image`).
        Configs are saved in the file `config.json` (for `timm_image` or `hf_text`)
        or  '<ckpt_name>.py' (for `mmdet_image`).

        Parameters
        ----------
        save_path : str
           Path to directory where models and configs should be saved.
        """
        return self._learner.dump_model(save_path=save_path)

    def export_onnx(
        self,
        data: Union[dict, pd.DataFrame],
        path: Optional[str] = None,
        batch_size: Optional[int] = None,
        verbose: Optional[bool] = False,
        opset_version: Optional[int] = 16,
        truncate_long_and_double: Optional[bool] = False,
    ):
        """
        Export this predictor's model to an ONNX file.

        When `path` argument is not provided, the method would not save the model into disk.
        Instead, it would export the onnx model into BytesIO and return its binary as bytes.

        Parameters
        ----------
        data
            Raw data used to trace and export the model.
            If this is None, will check if a processed batch is provided.
        path : str, default=None
            The export path of onnx model. If path is not provided, the method would export model to memory.
        batch_size
            The batch_size of export model's input.
            Normally the batch_size is a dynamic axis, so we could use a small value for faster export.
        verbose
            verbose flag in torch.onnx.export.
        opset_version
            opset_version flag in torch.onnx.export.
        truncate_long_and_double: bool, default False
            Truncate weights provided in int64 or double (float64) to int32 and float32

        Returns
        -------
        onnx_path : str or bytes
            A string that indicates location of the exported onnx model, if `path` argument is provided.
            Otherwise, would return the onnx model as bytes.
        """

        # Make sure _model is initialized
        self._learner.on_predict_start()

        return self._learner.export_onnx(
            data=data,
            path=path,
            batch_size=batch_size,
            verbose=verbose,
            opset_version=opset_version,
            truncate_long_and_double=truncate_long_and_double,
        )

    def optimize_for_inference(
        self,
        providers: Optional[Union[dict, List[str]]] = None,
    ):
        """
        Optimize the predictor's model for inference.

        Under the hood, the implementation would convert the PyTorch module into an ONNX module, so that
        we can leverage efficient execution providers in onnxruntime for faster inference.

        Parameters
        ----------
        providers : dict or str, default=None
            A list of execution providers for model prediction in onnxruntime.

            By default, the providers argument is None. The method would generate an ONNX module that
            would perform model inference with TensorrtExecutionProvider in onnxruntime, if tensorrt
            package is properly installed. Otherwise, the onnxruntime would fallback to use CUDA or CPU
            execution providers instead.

        Returns
        -------
        onnx_module : OnnxModule
            The onnx-based module that can be used to replace predictor._model for model inference.
        """
        return self._learner.optimize_for_inference(providers=providers)

    def fit_summary(self, verbosity=0, show_plot=False):
        """
        Output the training summary information from `fit()`.

        Parameters
        ----------
        verbosity : int, default = 2
            Verbosity levels range from 0 to 4 and control how much information is printed.
            verbosity = 0 for no output printing.
            TODO: Higher levels correspond to more detailed print statements
        show_plot : bool, default = False
            If True, shows the model summary plot in browser when verbosity > 1.

        Returns
        -------
        Dict containing various detailed information.
        We do not recommend directly printing this dict as it may be very large.
        """
        return self._learner.fit_summary(verbosity=verbosity, show_plot=show_plot)

    def list_supported_models(self, pretrained=True):
        """
        List supported models for each problem type.

        Parameters
        ----------
        pretrained : bool, default = True
            If True, only return the models with pretrained weights.
            If False, return all the models as long as there is model definition.

        Returns
        -------
        a list of model names
        """
        return self._learner.list_supported_models(pretrained=pretrained)
