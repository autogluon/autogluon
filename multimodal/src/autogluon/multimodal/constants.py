# Column/Label Types
NULL = "null"
CATEGORICAL = "categorical"
TEXT = "text"
NUMERICAL = "numerical"
IMAGE_PATH = "image_path"

# Problem types
CLASSIFICATION = "classification"
BINARY = "binary"
MULTICLASS = "multiclass"
REGRESSION = "regression"
FEW_SHOT = "few_shot"
DEFAULT_SHOT = "default_shot"
DEPRECATED_ZERO_SHOT = "zero_shot"

# Pipelines
FEATURE_EXTRACTION = "feature_extraction"
ZERO_SHOT_IMAGE_CLASSIFICATION = "zero_shot_image_classification"
OBJECT_DETECTION = "object_detection"
OCR_TEXT_DETECTION = "ocr_text_detection"
OCR_TEXT_RECOGNITION = "ocr_text_recognition"

# Input keys
IMAGE = "image"
IMAGE_VALID_NUM = "image_valid_num"
LABEL = "label"
TEXT_TOKEN_IDS = "text_token_ids"
CHOICES_IDS = "choices_ids"
TEXT_VALID_LENGTH = "text_valid_length"
TEXT_SEGMENT_IDS = "text_segment_ids"
COLUMN = "column"

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

# Metric
MAX = "max"
MIN = "min"
ACCURACY = "accuracy"
ACC = "acc"
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
METRIC_MODE_MAP = {
    ACC: MAX,
    ACCURACY: MAX,
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
}
VALID_METRICS = METRIC_MODE_MAP.keys()

# Training status
TRAIN = "train"
VAL = "val"
TEST = "test"
PREDICT = "predict"

# Model sources
HUGGINGFACE = "huggingface"
TIMM = "timm"
MMDET = "mmdet"
MMOCR = "mmocr"

# Modality keys. may need to update here if new modality keys are added in above.
ALL_MODALITIES = [IMAGE, TEXT, CATEGORICAL, NUMERICAL]

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

# registered model keys. TODO: document how to add new models.
CLIP = "clip"
TIMM_IMAGE = "timm_image"
HF_TEXT = "hf_text"
T_FEW = "t_few"
NUMERICAL_MLP = "numerical_mlp"
CATEGORICAL_MLP = "categorical_mlp"
NUMERICAL_TRANSFORMER = "numerical_transformer"
CATEGORICAL_TRANSFORMER = "categorical_transformer"
FUSION_MLP = "fusion_mlp"
FUSION_TRANSFORMER = "fusion_transformer"
MMDET_IMAGE = "mmdet_image"
MMOCR_TEXT_DET = "mmocr_text_detection"
MMOCR_TEXT_RECOG = "mmocr_text_recognition"
HF_MODELS = (HF_TEXT, T_FEW, CLIP)
MMCV_MODELS = (MMDET_IMAGE, MMOCR_TEXT_DET, MMOCR_TEXT_RECOG)

# metric learning loss type
CONTRASTIVE_LOSS = "contrastive_loss"

# metric learning distance type
COSINE_SIMILARITY = "cosine_similarity"

# metric learning miner type
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
