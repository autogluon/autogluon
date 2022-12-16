"""Problem types supported in MultiModalPredictor"""

from dataclasses import dataclass, field
from .registry import Registry
from .constants import (
    CLASSIFICATION,
    BINARY,
    MULTICLASS,
    REGRESSION,
    OBJECT_DETECTION,
    TEXT_SIMILARITY,
    IMAGE_SIMILARITY,
    IMAGE_TEXT_SIMILARITY,
    NER,
    NAMED_ENTITY_RECOGNITION,
    FEATURE_EXTRACTION,
    ZERO_SHOT_IMAGE_CLASSIFICATION,
    FEW_SHOT_TEXT_CLASSIFICATION,
    OCR_TEXT_DETECTION,
    OCR_TEXT_RECOGNITION,
    IMAGE,
    IMAGE_BYTEARRAY,
    TEXT,
    CATEGORICAL,
    NUMERICAL,
    ROIS,
    TEXT_NER,
    NER_ANNOTATION,
)
from typing import Optional, Set


problem_type_reg = Registry("problem_types")


@dataclass
class ProblemType:
    """Problem type class. Stores the name of the problem
    and its related properties. Some of the properties will be useful for the label / feature inference logic."""

    name: str  # Name of the problem
    support_fit: bool = True  # Whether the problem type support `.fit()`
    inference_ready: bool = False  # Support `.predict()` and `.evaluate()` without calling `.fit()`
    is_matching: bool = False  # Whether the problem belongs to the matching category
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
problem_type_reg.register(
    CLASSIFICATION,
    ProblemType(
        name=CLASSIFICATION,
        supported_modality_type={IMAGE, IMAGE_BYTEARRAY, TEXT, CATEGORICAL, NUMERICAL},
        supported_label_type={CATEGORICAL},
    ),
)
problem_type_reg.register(
    BINARY,
    ProblemType(
        name=BINARY,
        supported_modality_type={IMAGE, IMAGE_BYTEARRAY, TEXT, CATEGORICAL, NUMERICAL},
        supported_label_type={CATEGORICAL},
    ),
)
problem_type_reg.register(
    MULTICLASS,
    ProblemType(
        name=MULTICLASS,
        supported_modality_type={IMAGE, IMAGE_BYTEARRAY, TEXT, CATEGORICAL, NUMERICAL},
        supported_label_type={CATEGORICAL},
    ),
)

# Regression: Arbitrary combination of image, text, tabular data --> numeric value
problem_type_reg.register(
    REGRESSION,
    ProblemType(
        name=REGRESSION,
        supported_modality_type={IMAGE, IMAGE_BYTEARRAY, TEXT, CATEGORICAL, NUMERICAL},
        supported_label_type={NUMERICAL},
    ),
)

# Object detection: image --> bounding boxes
problem_type_reg.register(
    OBJECT_DETECTION,
    ProblemType(
        name=OBJECT_DETECTION,
        inference_ready=True,
        supported_modality_type={IMAGE},
        supported_label_type={ROIS},
        force_exist_modality={IMAGE},
    ),
)

# Matching: text <--> text, image <--> image, text <--> image
problem_type_reg.register(
    TEXT_SIMILARITY,
    ProblemType(
        name=TEXT_SIMILARITY,
        inference_ready=True,
        is_matching=True,
        supported_modality_type={TEXT},
        supported_label_type={CATEGORICAL, NUMERICAL},
        force_exist_modality={TEXT},
    ),
)
problem_type_reg.register(
    IMAGE_SIMILARITY,
    ProblemType(
        name=IMAGE_SIMILARITY,
        inference_ready=True,
        is_matching=True,
        supported_modality_type={IMAGE},
        supported_label_type={CATEGORICAL, NUMERICAL},
        force_exist_modality={IMAGE},
    ),
)
problem_type_reg.register(
    IMAGE_TEXT_SIMILARITY,
    ProblemType(
        name=IMAGE_TEXT_SIMILARITY,
        inference_ready=True,
        is_matching=True,
        supported_modality_type={IMAGE, TEXT},
        supported_label_type={CATEGORICAL, NUMERICAL},
        force_exist_modality={IMAGE, TEXT},
    ),
)

# Entity Extraction: text (tied to the entity), [other text], [image], [tabular] --> entity
_ner_type = ProblemType(
    name=NER,
    supported_modality_type={IMAGE, TEXT, CATEGORICAL, NUMERICAL, TEXT_NER},
    supported_label_type={NER_ANNOTATION},
    force_exist_modality={TEXT_NER},
)
problem_type_reg.register(NER, _ner_type),
problem_type_reg.register(NAMED_ENTITY_RECOGNITION, _ner_type),

# Feature Extraction: text --> feature, image --> features
problem_type_reg.register(
    FEATURE_EXTRACTION,
    ProblemType(
        name=FEATURE_EXTRACTION, support_fit=False, inference_ready=True, supported_modality_type={IMAGE, TEXT}
    ),
)

# Zero-shot Image classification
problem_type_reg.register(
    ZERO_SHOT_IMAGE_CLASSIFICATION,
    ProblemType(
        name=ZERO_SHOT_IMAGE_CLASSIFICATION,
        support_fit=False,
        inference_ready=True,
        supported_modality_type={IMAGE},
        force_exist_modality={IMAGE},
    ),
)

# Few-shot Text classification. TODO: For few-shot problems, they may be revised to be presets
problem_type_reg.register(
    FEW_SHOT_TEXT_CLASSIFICATION,
    ProblemType(
        name=FEW_SHOT_TEXT_CLASSIFICATION,
        support_fit=True,
        inference_ready=False,
        experimental=True,
        supported_modality_type={TEXT},
        force_exist_modality={TEXT},
    ),
)

# OCR. TODO: Improve the definition of OCR.
problem_type_reg.register(
    OCR_TEXT_DETECTION,
    ProblemType(
        name=OCR_TEXT_DETECTION,
        support_fit=False,
        inference_ready=True,
        experimental=True,
        supported_modality_type={IMAGE},
    ),
)

problem_type_reg.register(
    OCR_TEXT_RECOGNITION,
    ProblemType(
        name=OCR_TEXT_RECOGNITION,
        support_fit=False,
        inference_ready=True,
        experimental=True,
        supported_modality_type={IMAGE},
    ),
)
