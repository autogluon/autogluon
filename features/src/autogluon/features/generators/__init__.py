from .abstract import AbstractFeatureGenerator
from .arithmetic.preprocessor import ArithmeticFeatureGenerator
from .astype import AsTypeFeatureGenerator
from .auto_ml_pipeline import AutoMLInterpretablePipelineFeatureGenerator, AutoMLPipelineFeatureGenerator
from .binned import BinnedFeatureGenerator
from .bulk import BulkFeatureGenerator
from .cat_as_num import CatAsNumFeatureGenerator
from .cat_int import CategoricalInteractionFeatureGenerator
from .category import CategoryFeatureGenerator
from .datetime import DatetimeFeatureGenerator
from .drop_duplicates import DropDuplicatesFeatureGenerator
from .drop_unique import DropUniqueFeatureGenerator
from .dummy import DummyFeatureGenerator
from .fillna import FillNaFeatureGenerator
from .frequency import FrequencyFeatureGenerator
from .identity import IdentityFeatureGenerator
from .isnan import IsNanFeatureGenerator
from .label_encoder import LabelEncoderFeatureGenerator
from .memory_minimize import CategoryMemoryMinimizeFeatureGenerator, NumericMemoryMinimizeFeatureGenerator
from .one_hot_encoder import OneHotEncoderFeatureGenerator
from .oof_target_encoder import OOFTargetEncodingFeatureGenerator
from .pipeline import PipelineFeatureGenerator
from .rename import RenameFeatureGenerator
from .text_ngram import TextNgramFeatureGenerator
from .text_special import TextSpecialFeatureGenerator

REGISTERED_FE_CLS_LST = [
    AbstractFeatureGenerator,
    ArithmeticFeatureGenerator,
    AsTypeFeatureGenerator,
    AutoMLInterpretablePipelineFeatureGenerator,
    AutoMLPipelineFeatureGenerator,
    BinnedFeatureGenerator,
    BulkFeatureGenerator,
    CatAsNumFeatureGenerator,
    CategoricalInteractionFeatureGenerator,
    CategoryFeatureGenerator,
    DatetimeFeatureGenerator,
    DropDuplicatesFeatureGenerator,
    DropUniqueFeatureGenerator,
    DummyFeatureGenerator,
    FillNaFeatureGenerator,
    FrequencyFeatureGenerator,
    IdentityFeatureGenerator,
    IsNanFeatureGenerator,
    LabelEncoderFeatureGenerator,
    CategoryMemoryMinimizeFeatureGenerator,
    NumericMemoryMinimizeFeatureGenerator,
    OneHotEncoderFeatureGenerator,
    OOFTargetEncodingFeatureGenerator,
    PipelineFeatureGenerator,
    RenameFeatureGenerator,
    TextNgramFeatureGenerator,
    TextSpecialFeatureGenerator,
]
