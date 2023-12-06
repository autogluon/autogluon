"""Storing the constants"""
from autogluon.core.metrics import METRICS

# Column/Label Types
NULL = "null"
CATEGORICAL = "categorical"
TEXT = "text"
TEXT_NER = "text_ner"  # Added for NER text column
NUMERICAL = "numerical"
IMAGE_PATH = "image_path"
IMAGE_BYTEARRAY = "image_bytearray"
IDENTIFIER = "identifier"
DOCUMENT = "document"
DOCUMENT_IMAGE = "document_image"
DOCUMENT_PDF = "document_pdf"

# Scarcity modes
FEW_SHOT = "few_shot"
DEFAULT_SHOT = "default_shot"
ZERO_SHOT = "zero_shot"

# Problem types
CLASSIFICATION = "classification"
BINARY = "binary"
MULTICLASS = "multiclass"
REGRESSION = "regression"
NER = "ner"
NAMED_ENTITY_RECOGNITION = "named_entity_recognition"
FEATURE_EXTRACTION = "feature_extraction"
ZERO_SHOT_IMAGE_CLASSIFICATION = "zero_shot_image_classification"
OBJECT_DETECTION = "object_detection"
OPEN_VOCABULARY_OBJECT_DETECTION = "open_vocabulary_object_detection"
OCR = "ocr"
OCR_TEXT_DETECTION = f"{OCR}_text_detection"
OCR_TEXT_RECOGNITION = f"{OCR}_text_recognition"
IMAGE_SIMILARITY = "image_similarity"
TEXT_SIMILARITY = "text_similarity"
IMAGE_TEXT_SIMILARITY = "image_text_similarity"
FEW_SHOT_CLASSIFICATION = "few_shot_classification"
SEMANTIC_SEGMENTATION = "semantic_segmentation"

# Input keys
IMAGE = "image"
IMAGE_META = "image_meta"
IMAGE_VALID_NUM = "image_valid_num"
LABEL = "label"
TEXT_TOKEN_IDS = "text_token_ids"
CHOICES_IDS = "choices_ids"
TEXT_VALID_LENGTH = "text_valid_length"
TEXT_SEGMENT_IDS = "text_segment_ids"
COLUMN = "column"
ATTENTION_MASK = "attention_mask"
TOKEN_TYPE_IDS = "token_type_ids"
PIXEL_VALUES = "pixel_values"
INPUT_IDS = "input_ids"
SEMANTIC_SEGMENTATION_IMG = "semantic_segmentation_img"
SEMANTIC_SEGMENTATION_GT = "semantic_segmentation_gt"

# Output keys
LOGITS = "logits"
TEMPLATE_LOGITS = "template_logits"
LM_TARGET = "lm_target"
LOSS = "loss"
OUTPUT = "output"
WEIGHT = "weight"
FEATURES = "features"
RAW_FEATURES = "raw_features"
MASKS = "masks"
PROBABILITY = "probability"
COLUMN_FEATURES = "column_features"
BBOX = "bbox"
ROIS = "rois"
SCORE = "score"
LOGIT_SCALE = "logit_scale"

# Metric for Object Detection
MAP = "map"
MEAN_AVERAGE_PRECISION = "mean_average_precision"
MAP_50 = "map_50"
MAP_75 = "map_75"
MAP_SMALL = "map_small"
MAP_MEDIUM = "map_medium"
MAP_LARGE = "map_large"
MAR_1 = "mar_1"
MAR_10 = "mar_10"
MAR_100 = "mar_100"
MAR_SMALL = "mar_small"
MAR_MEDIUM = "mar_medium"
MAR_LARGE = "mar_large"
DETECTION_METRICS = [
    MAP,
    MEAN_AVERAGE_PRECISION,
    MAP_50,
    MAP_75,
    MAP_SMALL,
    MAP_MEDIUM,
    MAP_LARGE,
    MAR_1,
    MAR_10,
    MAR_100,
    MAR_SMALL,
    MAR_MEDIUM,
    MAR_LARGE,
]

# Metric
MAX = "max"
MIN = "min"
ACCURACY = "accuracy"
ACC = "acc"
OVERALL_ACCURACY = "overall_accuracy"
RMSE = "rmse"
ROOT_MEAN_SQUARED_ERROR = "root_mean_squared_error"
R2 = "r2"
PEARSONR = "pearsonr"
SPEARMANR = "spearmanr"
QUADRATIC_KAPPA = "quadratic_kappa"
ROC_AUC = "roc_auc"
AVERAGE_PRECISION = "average_precision"
LOG_LOSS = "log_loss"
CROSS_ENTROPY = "cross_entropy"
COSINE_EMBEDDING_LOSS = "cosine_embedding_loss"
F1 = "f1"
OVERALL_F1 = "overall_f1"
F1_MACRO = "f1_macro"
F1_MICRO = "f1_micro"
F1_WEIGHTED = "f1_weighted"
NER_TOKEN_F1 = "ner_token_f1"
DIRECT_LOSS = "direct_loss"
HIT_RATE = "hit_rate"
NDCG = "ndcg"
PRECISION = "precision"
RECALL = "recall"
MRR = "mrr"
SM = "sm"
EM = "em"
FM = "fm"
MAE = "mae"
BER = "ber"
IOU = "iou"
RETRIEVAL_METRICS = [NDCG, PRECISION, RECALL, MRR]
METRIC_MODE_MAP = {
    ACC: MAX,
    ACCURACY: MAX,
    DIRECT_LOSS: MIN,
    RMSE: MIN,
    ROOT_MEAN_SQUARED_ERROR: MIN,
    R2: MAX,
    QUADRATIC_KAPPA: MAX,
    ROC_AUC: MAX,
    LOG_LOSS: MIN,
    CROSS_ENTROPY: MIN,
    PEARSONR: MAX,
    SPEARMANR: MAX,
    F1: MAX,
    F1_MACRO: MAX,
    F1_MICRO: MAX,
    F1_WEIGHTED: MAX,
    MAP: MAX,
    MEAN_AVERAGE_PRECISION: MAX,
    NER_TOKEN_F1: MAX,
    OVERALL_F1: MAX,
    RECALL: MAX,
    SM: MAX,
    IOU: MAX,
    BER: MIN,
}

MATCHING_METRICS = {
    BINARY: [ROC_AUC, ROC_AUC],
    MULTICLASS: [SPEARMANR, SPEARMANR],
    REGRESSION: [SPEARMANR, SPEARMANR],
}
MATCHING_METRICS_WITHOUT_PROBLEM_TYPE = [RECALL, NDCG]

# Training status
TRAIN = "train"
VALIDATE = "validate"
TEST = "test"
PREDICT = "predict"

# Model sources
HUGGINGFACE = "huggingface"
TIMM = "timm"
MMDET = "mmdet"
MMOCR = "mmocr"

# Modality keys. may need to update here if new modality keys are added in above.
ALL_MODALITIES = [IMAGE, TEXT, CATEGORICAL, NUMERICAL, TEXT_NER, DOCUMENT, SEMANTIC_SEGMENTATION_IMG]

# Keys to compute metrics
Y_PRED = "y_pred"
Y_PRED_PROB = "y_pred_prob"
Y_TRUE = "y_true"

# Configuration keys
MODEL = "model"
DATA = "data"
OPTIMIZATION = "optimization"
ENVIRONMENT = "environment"
DISTILLER = "distiller"
MATCHER = "matcher"
VALID_CONFIG_KEYS = [MODEL, DATA, OPTIMIZATION, ENVIRONMENT, DISTILLER, MATCHER]

# Image normalization mean and std. This is only to normalize images for the CLIP model.
CLIP_IMAGE_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_IMAGE_STD = (0.26862954, 0.26130258, 0.27577711)

# Logger name
AUTOMM = "automm"

# environment variables
AUTOMM_TUTORIAL_MODE = "AUTOMM_TUTORIAL_MODE"

# error try
GET_ITEM_ERROR_RETRY = 50

# top-k checkpoint average methods
UNIFORM_SOUP = "uniform_soup"
GREEDY_SOUP = "greedy_soup"
BEST = "best"

# efficient finetuning strategies
NORM_FIT = "norm_fit"
BIT_FIT = "bit_fit"
LORA = "lora"
LORA_BIAS = "lora_bias"
LORA_NORM = "lora_norm"
IA3 = "ia3"
IA3_BIAS = "ia3_bias"
IA3_NORM = "ia3_norm"
IA3_LORA = "ia3_lora"
IA3_LORA_BIAS = "ia3_lora_bias"
IA3_LORA_NORM = "Ia3_lora_norm"
PEFT_STRATEGIES = [
    BIT_FIT,
    NORM_FIT,
    LORA,
    LORA_BIAS,
    LORA_NORM,
    IA3,
    IA3_BIAS,
    IA3_NORM,
    IA3_LORA,
    IA3_LORA_BIAS,
    IA3_LORA_NORM,
]

# DeepSpeed constants
DEEPSPEED_OFFLOADING = "deepspeed_stage_3_offload"
DEEPSPEED_STRATEGY = "deepspeed"
DEEPSPEED_MODULE = "autogluon.multimodal.optimization.deepspeed"
DEEPSPEED_MIN_PL_VERSION = "1.7.1"

# registered model keys. TODO: document how to add new models.
CLIP = "clip"
TIMM_IMAGE = "timm_image"
HF_TEXT = "hf_text"
T_FEW = "t_few"
NUMERICAL_MLP = "numerical_mlp"
CATEGORICAL_MLP = "categorical_mlp"
FUSION = "fusion"
FUSION_MLP = f"{FUSION}_mlp"
FUSION_TRANSFORMER = f"{FUSION}_transformer"
FT_TRANSFORMER = "ft_transformer"
FUSION_NER = f"{FUSION}_{NER}"
MMDET_IMAGE = "mmdet_image"
MMOCR_TEXT_DET = "mmocr_text_detection"
MMOCR_TEXT_RECOG = "mmocr_text_recognition"
OVD = "ovd"
NER_TEXT = "ner_text"
DOCUMENT_TRANSFORMER = "document_transformer"
HF_MODELS = (HF_TEXT, T_FEW, CLIP, NER_TEXT, DOCUMENT_TRANSFORMER)
MMLAB_MODELS = (MMDET_IMAGE, MMOCR_TEXT_DET, MMOCR_TEXT_RECOG)
SAM = "sam"

# matcher loss type
CONTRASTIVE_LOSS = "contrastive_loss"
MULTI_NEGATIVES_SOFTMAX_LOSS = "multi_negatives_softmax_loss"

# matcher distance type
COSINE_SIMILARITY = "cosine_similarity"

# matcher miner type
PAIR_MARGIN_MINER = "pair_margin_miner"

# checkpoints
RAY_TUNE_CHECKPOINT = "ray_tune_checkpoint.ckpt"
BEST_K_MODELS_FILE = "best_k_models.yaml"
LAST_CHECKPOINT = "last.ckpt"
MODEL_CHECKPOINT = "model.ckpt"

# url
S3_PREFIX = "s3://"
SOURCEPROMPT_URL = "https://automl-mm-bench.s3.amazonaws.com/few_shot/templates.zip"
SOURCEPROMPT_SHA1 = "c25cdf3730ff96ab4859b72e18d46ff117b62bd6"

# ner
ENTITY_GROUP = "entity_group"
START_OFFSET = "start"
END_OFFSET = "end"
TOKEN_WORD_MAPPING = "token_word_mapping"
WORD_OFFSETS = "word_offsets"
NER_RET = "ner_ret"
NER_ANNOTATION = "ner_annotation"

# matcher
QUERY = "query"
RESPONSE = "response"
QUERY_RESPONSE = f"{QUERY}_{RESPONSE}"
PAIR = "pair"
TRIPLET = "triplet"

# mmdet
XYWH = "xywh"
XYXY = "xyxy"
BBOX_FORMATS = [XYWH, XYXY]

# open vocabulary detection
PROMPT = "prompt"
OVD_RET = "ovd_ret"

# sam (multi-class)
CLASS_LOGITS = "class_logits"
MASK_LABEL = "mask_label"
CLASS_LABEL = "class_label"
SEMANTIC_MASK = "semantic_mask"

# presets
DEFAULT = "default"
HIGH_QUALITY = "high_quality"
MEDIUM_QUALITY = "medium_quality"
BEST_QUALITY = "best_quality"
ALL_MODEL_QUALITIES = [HIGH_QUALITY, MEDIUM_QUALITY, BEST_QUALITY, DEFAULT]

# datasets
DEFAULT_DATASET = "default_dataset"
MULTI_IMAGE_MIX_DATASET = "multi_image_mix_dataset"

# strategies
DDP = "ddp"
DDP_FIND_UNUSED_PARAMETERS_FALSE = "ddp_find_unused_parameters_false"
DDP_FIND_UNUSED_PARAMETERS_TRUE = "ddp_find_unused_parameters_true"
DDP_STRATEGIES = [DDP, DDP_FIND_UNUSED_PARAMETERS_FALSE, DDP_FIND_UNUSED_PARAMETERS_TRUE]

# torch constants
TORCH_COMPILE_MIN_VERSION = "2.2.0.dev20230908"
