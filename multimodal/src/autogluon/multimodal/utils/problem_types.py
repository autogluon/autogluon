"""Problem types supported in MultiModalPredictor"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Set

from ..constants import (
    ACCURACY,
    BINARY,
    CATEGORICAL,
    CLASSIFICATION,
    EVALUATION_METRICS,
    FEATURE_EXTRACTION,
    FEW_SHOT_CLASSIFICATION,
    IMAGE,
    IMAGE_BASE64_STR,
    IMAGE_BYTEARRAY,
    IMAGE_SIMILARITY,
    IMAGE_TEXT_SIMILARITY,
    IOU,
    MAP,
    MULTICLASS,
    NAMED_ENTITY_RECOGNITION,
    NER,
    NER_ANNOTATION,
    NER_TOKEN_F1,
    NUMERICAL,
    OBJECT_DETECTION,
    REGRESSION,
    RMSE,
    ROC_AUC,
    ROIS,
    SEMANTIC_SEGMENTATION,
    TEXT,
    TEXT_NER,
    TEXT_SIMILARITY,
    VALIDATION_METRICS,
    ZERO_SHOT_IMAGE_CLASSIFICATION,
)
from .registry import Registry

logger = logging.getLogger(__name__)

PROBLEM_TYPES_REG = Registry("problem_type_properties")


@dataclass
class ProblemTypeProperty:
    """
    Property of the problem. Stores the name of the problem
    and its related properties. Some properties are used in the label / feature inference logic.
    """

    name: str  # Name of the problem
    support_fit: bool = True  # Whether the problem type support `.fit()`
    support_zero_shot: bool = False  # Support `.predict()` and `.evaluate()` without calling `.fit()`
    is_matching: bool = False  # Whether the problem belongs to the matching category
    is_classification: bool = False  # Whether the problem is a classification problem
    experimental: bool = False  # Indicate whether the problem is experimental

    # The collection of modality types the problem supports.
    # Multiple column types may be parsed into the same modality. For example
    #   IMAGE, IMAGE_PATH, IMAGE_BYTEARRAY, IMAGE_BASE64_STR --> IMAGE
    # It will be used to analyze the dataframe and detect the columns.
    supported_modality_type: Set[str] = field(default_factory=set)

    # The collection of label column types the problem supports.
    supported_label_type: Optional[Set[str]] = None

    # The modalities that have to appear in the table.
    force_exist_modality: Optional[Set[str]] = None

    # The fallback label type of the problem
    _fallback_label_type: Optional[str] = None

    # If evaluations are supported
    _support_eval: Optional[bool] = False

    # Overwrite the default setting (first in supported_evaluation_metrics)
    _fallback_evaluation_metric: Optional[str] = None

    # The validation metric fallback
    # It may be different from the evaluation metric
    _fallback_validation_metric: Optional[str] = None

    @property
    def fallback_label_type(self):
        if self._fallback_label_type is None and len(self.supported_label_type) == 1:
            return next(iter(self.supported_label_type))
        else:
            return self._fallback_label_type

    @property
    def supported_evaluation_metrics(self):
        if self._support_eval:
            return EVALUATION_METRICS[self.name]
        else:
            return []

    @property
    def fallback_evaluation_metric(self):
        if self._fallback_evaluation_metric:
            return self._fallback_evaluation_metric
        elif self._support_eval:
            return self.supported_evaluation_metrics[0]
        else:
            return None

    @property
    def supported_validation_metrics(self):
        if self._support_eval:
            return VALIDATION_METRICS[self.name]
        else:
            return []

    @property
    def fallback_validation_metric(self):
        if self._fallback_validation_metric:
            assert self._fallback_validation_metric in self.supported_validation_metrics
            return self._fallback_validation_metric
        else:
            return None


# Classification: Arbitrary combination of image, text, tabular data --> categorical value
PROBLEM_TYPES_REG.register(
    CLASSIFICATION,
    ProblemTypeProperty(
        name=CLASSIFICATION,
        supported_modality_type={IMAGE, IMAGE_BYTEARRAY, IMAGE_BASE64_STR, TEXT, CATEGORICAL, NUMERICAL},
        supported_label_type={CATEGORICAL},
        is_classification=True,
    ),
)
PROBLEM_TYPES_REG.register(
    BINARY,
    ProblemTypeProperty(
        name=BINARY,
        supported_modality_type={IMAGE, IMAGE_BYTEARRAY, IMAGE_BASE64_STR, TEXT, CATEGORICAL, NUMERICAL},
        supported_label_type={CATEGORICAL},
        is_classification=True,
        _support_eval=True,
        _fallback_evaluation_metric=ROC_AUC,
        _fallback_validation_metric=ROC_AUC,
    ),
)
PROBLEM_TYPES_REG.register(
    MULTICLASS,
    ProblemTypeProperty(
        name=MULTICLASS,
        supported_modality_type={IMAGE, IMAGE_BYTEARRAY, IMAGE_BASE64_STR, TEXT, CATEGORICAL, NUMERICAL},
        supported_label_type={CATEGORICAL},
        is_classification=True,
        _support_eval=True,
        _fallback_evaluation_metric=ACCURACY,
        _fallback_validation_metric=ACCURACY,
    ),
)

# Regression: Arbitrary combination of image, text, tabular data --> numeric value
PROBLEM_TYPES_REG.register(
    REGRESSION,
    ProblemTypeProperty(
        name=REGRESSION,
        supported_modality_type={IMAGE, IMAGE_BYTEARRAY, IMAGE_BASE64_STR, TEXT, CATEGORICAL, NUMERICAL},
        supported_label_type={NUMERICAL},
        _support_eval=True,
        _fallback_evaluation_metric=RMSE,
        _fallback_validation_metric=RMSE,
    ),
)

# Object detection: image --> bounding boxes
PROBLEM_TYPES_REG.register(
    OBJECT_DETECTION,
    ProblemTypeProperty(
        name=OBJECT_DETECTION,
        support_zero_shot=True,
        supported_modality_type={IMAGE},
        supported_label_type={ROIS},
        force_exist_modality={IMAGE},
        _support_eval=True,
        _fallback_validation_metric=MAP,
    ),
)

# Real-World Semantic Segmentation: image --> image
PROBLEM_TYPES_REG.register(
    SEMANTIC_SEGMENTATION,
    ProblemTypeProperty(
        name=SEMANTIC_SEGMENTATION,
        support_zero_shot=True,
        support_fit=True,
        supported_modality_type={IMAGE},
        supported_label_type={IMAGE},
        force_exist_modality={IMAGE},
        _support_eval=True,
        _fallback_evaluation_metric=IOU,
        _fallback_validation_metric=IOU,
    ),
)

# Matching: text <--> text, image <--> image, text <--> image
PROBLEM_TYPES_REG.register(
    TEXT_SIMILARITY,
    ProblemTypeProperty(
        name=TEXT_SIMILARITY,
        support_zero_shot=True,
        is_matching=True,
        supported_modality_type={TEXT},
        supported_label_type={CATEGORICAL, NUMERICAL},
        force_exist_modality={TEXT},
    ),
)
PROBLEM_TYPES_REG.register(
    IMAGE_SIMILARITY,
    ProblemTypeProperty(
        name=IMAGE_SIMILARITY,
        support_zero_shot=True,
        is_matching=True,
        supported_modality_type={IMAGE},
        supported_label_type={CATEGORICAL, NUMERICAL},
        force_exist_modality={IMAGE},
    ),
)
PROBLEM_TYPES_REG.register(
    IMAGE_TEXT_SIMILARITY,
    ProblemTypeProperty(
        name=IMAGE_TEXT_SIMILARITY,
        support_zero_shot=True,
        is_matching=True,
        supported_modality_type={IMAGE, TEXT},
        supported_label_type={CATEGORICAL, NUMERICAL},
        force_exist_modality={IMAGE, TEXT},
    ),
)

# Entity Extraction: text (tied to the entity), [other text], [image], [tabular] --> entity
_ner_property = ProblemTypeProperty(
    name=NER,
    supported_modality_type={IMAGE, TEXT, CATEGORICAL, NUMERICAL, TEXT_NER},
    supported_label_type={NER_ANNOTATION},
    force_exist_modality={TEXT_NER},
    _support_eval=True,
    _fallback_validation_metric=NER_TOKEN_F1,
)
PROBLEM_TYPES_REG.register(NER, _ner_property)
PROBLEM_TYPES_REG.register(NAMED_ENTITY_RECOGNITION, _ner_property)

# Feature Extraction: text --> feature, image --> features
PROBLEM_TYPES_REG.register(
    FEATURE_EXTRACTION,
    ProblemTypeProperty(
        name=FEATURE_EXTRACTION, support_fit=False, support_zero_shot=True, supported_modality_type={IMAGE, TEXT}
    ),
)

# Zero-shot Image classification
PROBLEM_TYPES_REG.register(
    ZERO_SHOT_IMAGE_CLASSIFICATION,
    ProblemTypeProperty(
        name=ZERO_SHOT_IMAGE_CLASSIFICATION,
        support_fit=False,
        support_zero_shot=True,
        supported_modality_type={IMAGE},
        force_exist_modality={IMAGE},
    ),
)

PROBLEM_TYPES_REG.register(
    FEW_SHOT_CLASSIFICATION,
    ProblemTypeProperty(
        name=FEW_SHOT_CLASSIFICATION,
        support_fit=True,
        support_zero_shot=False,
        supported_modality_type={IMAGE, TEXT},
        supported_label_type={CATEGORICAL},
        _support_eval=True,
        _fallback_evaluation_metric=ACCURACY,
        _fallback_validation_metric=ACCURACY,
    ),
)


def infer_problem_type_by_eval_metric(eval_metric_name: str, problem_type: str):
    if eval_metric_name is not None and eval_metric_name.lower() in [
        "rmse",
        "r2",
        "pearsonr",
        "spearmanr",
    ]:
        if problem_type is None:
            logger.debug(
                f"Infer problem type to be a regression problem "
                f"since the evaluation metric is set as {eval_metric_name}."
            )
            problem_type = REGRESSION
        else:
            problem_prop = PROBLEM_TYPES_REG.get(problem_type)
            if NUMERICAL not in problem_prop.supported_label_type:
                raise ValueError(
                    f"The provided evaluation metric will require the problem "
                    f"to support label type = {NUMERICAL}. However, "
                    f"the provided problem type = {problem_type} only "
                    f"supports label type = {problem_prop.supported_label_type}."
                )

    return problem_type
