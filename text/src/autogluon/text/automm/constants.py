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

# Output keys
LOGITS = "logits"
LOSS = "loss"
OUTPUT = "output"
WEIGHT = "weight"
FEATURES = "features"

# Metric
MAX = "max"
MIN = "min"
ACCURACY = "accuracy"
ACC = "acc"
RMSE = "rmse"
R2 = "r2"
QUADRATIC_KAPPA = "quadratic_kappa"

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

# Image normalization mean and std. This is only to normalize images for the CLIP model.
CLIP_IMAGE_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_IMAGE_STD = (0.26862954, 0.26130258, 0.27577711)

# Logger name
AUTOMM = "automm"

# environment variables
AUTOMM_TUTORIAL_MODE = "AUTOMM_TUTORIAL_MODE"

# error try
GET_ITEM_ERROR_RETRY = 50

# mini-ensemble methods
UNION_SOUP = 'union_soup'
GREEDY_SOUP = 'greedy_soup'
BEST_SOUP = 'best_soup'

# registered model keys. TODO: document how to add new models.
CLIP = "clip"
TIMM_IMAGE = "timm_image"
HF_TEXT = "hf_text"
NUMERICAL_MLP = "numerical_mlp"
CATEGORICAL_MLP = "categorical_mlp"
FUSION_MLP = "fusion_mlp"
