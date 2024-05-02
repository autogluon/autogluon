"""Problem types supported in MultiModalPredictor"""

from dataclasses import dataclass, field
from typing import Optional, Set

from .constants import (
    BINARY,
    CATEGORICAL,
    CLASSIFICATION,
    FEATURE_EXTRACTION,
    FEW_SHOT_TEXT_CLASSIFICATION,
    IMAGE,
    IMAGE_BYTEARRAY,
    IMAGE_SIMILARITY,
    IMAGE_TEXT_SIMILARITY,
    MULTICLASS,
    NAMED_ENTITY_RECOGNITION,
    NER,
    NER_ANNOTATION,
    NUMERICAL,
    OBJECT_DETECTION,
    OCR_TEXT_DETECTION,
    OCR_TEXT_RECOGNITION,
    REGRESSION,
    ROIS,
    TEXT,
    TEXT_NER,
    TEXT_SIMILARITY,
    ZERO_SHOT_IMAGE_CLASSIFICATION,
)
from .registry import Registry

PROBLEM_TYPES_REG = Registry("problem_type_properties")


@dataclass
class ProblemTypeProperty:
    """Property of the problem. Stores the name of the problem
    and its related properties. Some properties are used in the label / feature inference logic."""

    name: str  # Name of the problem
    support_fit: bool = True  # Whether the problem type support `.fit()`
    support_zero_shot: bool = False  # Support `.predict()` and `.evaluate()` without calling `.fit()`
    is_matching: bool = False  # Whether the problem belongs to the matching category
    is_classification: bool = False  # Whether the problem is a classification problem
    experimental: bool = False  # Indicate whether the problem is experimental

    # The collection of modality types the problem supports.
    # Multiple column types may be parsed into the same modality. For example
    #   IMAGE, IMAGE_PATH, IMAGE_BYTEARRAY --> IMAGE
    # It will be used to analyze the dataframe and detect the columns.
    supported_modality_type: Set[str] = field(default_factory=set)

    # The collection of label column types the problem supports.
    supported_label_type: Optional[Set[str]] = None

    # The modalities that have to appear in the table.
    force_exist_modality: Optional[Set[str]] = None

    # The fallback label type of the problem
    _fallback_label_type: Optional[str] = None

    @property
    def fallback_label_type(self):
        if self._fallback_label_type is None and len(self.supported_label_type) == 1:
            return next(iter(self.supported_label_type))
        else:
            return self._fallback_label_type


# Classification: Arbitrary combination of image, text, tabular data --> categorical value
PROBLEM_TYPES_REG.register(
    CLASSIFICATION,
    ProblemTypeProperty(
        name=CLASSIFICATION,
        supported_modality_type={IMAGE, IMAGE_BYTEARRAY, TEXT, CATEGORICAL, NUMERICAL},
        supported_label_type={CATEGORICAL},
        is_classification=True,
    ),
)
PROBLEM_TYPES_REG.register(
    BINARY,
    ProblemTypeProperty(
        name=BINARY,
        supported_modality_type={IMAGE, IMAGE_BYTEARRAY, TEXT, CATEGORICAL, NUMERICAL},
        supported_label_type={CATEGORICAL},
        is_classification=True,
    ),
)
PROBLEM_TYPES_REG.register(
    MULTICLASS,
    ProblemTypeProperty(
        name=MULTICLASS,
        supported_modality_type={IMAGE, IMAGE_BYTEARRAY, TEXT, CATEGORICAL, NUMERICAL},
        supported_label_type={CATEGORICAL},
        is_classification=True,
    ),
)

# Regression: Arbitrary combination of image, text, tabular data --> numeric value
PROBLEM_TYPES_REG.register(
    REGRESSION,
    ProblemTypeProperty(
        name=REGRESSION,
        supported_modality_type={IMAGE, IMAGE_BYTEARRAY, TEXT, CATEGORICAL, NUMERICAL},
        supported_label_type={NUMERICAL},
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
)
PROBLEM_TYPES_REG.register(NER, _ner_property),
PROBLEM_TYPES_REG.register(NAMED_ENTITY_RECOGNITION, _ner_property),

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

# Few-shot Text classification. TODO: For few-shot problems, they may be revised to be presets
PROBLEM_TYPES_REG.register(
    FEW_SHOT_TEXT_CLASSIFICATION,
    ProblemTypeProperty(
        name=FEW_SHOT_TEXT_CLASSIFICATION,
        support_fit=True,
        support_zero_shot=False,
        experimental=True,
        supported_modality_type={TEXT},
        force_exist_modality={TEXT},
    ),
)

# OCR. TODO: Improve the definition of OCR.
PROBLEM_TYPES_REG.register(
    OCR_TEXT_DETECTION,
    ProblemTypeProperty(
        name=OCR_TEXT_DETECTION,
        support_fit=False,
        support_zero_shot=True,
        experimental=True,
        supported_modality_type={IMAGE},
    ),
)

PROBLEM_TYPES_REG.register(
    OCR_TEXT_RECOGNITION,
    ProblemTypeProperty(
        name=OCR_TEXT_RECOGNITION,
        support_fit=False,
        support_zero_shot=True,
        experimental=True,
        supported_modality_type={IMAGE},
    ),
)
