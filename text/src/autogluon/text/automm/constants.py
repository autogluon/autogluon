# Column/Label Types
NULL = "null"
CATEGORICAL = "categorical"
TEXT = "text"
NUMERICAL = "numerical"
IMAGE_PATH = "image_path"

# Problem types
BINARY = "binary"
MULTICLASS = "multiclass"
REGRESSION = "regression"

# Input keys
IMAGE = "image"
IMAGE_VALID_NUM = "image_valid_num"
LABEL = "label"
TEXT_TOKEN_IDS = "text_token_ids"
TEXT_VALID_LENGTH = "text_valid_length"
TEXT_SEGMENT_IDS = "text_segment_ids"
COLUMN = "column"

# Output keys
LOGITS = "logits"
LOSS = "loss"
OUTPUT = "output"
WEIGHT = "weight"
FEATURES = "features"
MASK = "mask"

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

# Training status
TRAIN = "train"
VAL = "val"
TEST = "test"
PREDICT = "predict"

# Model sources
HUGGINGFACE = "huggingface"
TIMM = "timm"

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

# registered model keys. TODO: document how to add new models.
CLIP = "clip"
TIMM_IMAGE = "timm_image"
HF_TEXT = "hf_text"
NUMERICAL_MLP = "numerical_mlp"
CATEGORICAL_MLP = "categorical_mlp"
NUMERICAL_TRANSFORMER = "numerical_transformer"
CATEGORICAL_TRANSFORMER = "categorical_transformer"
FUSION_MLP = "fusion_mlp"
FUSION_TRANSFORMER = "fusion_transformer"
