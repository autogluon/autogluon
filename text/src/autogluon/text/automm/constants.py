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

# Log keys
VAL_LOSS = "val_loss"
VAL_ACC = "val_acc"
TEST_ACC = "test_acc"
MODE = "mode"
MAX = "max"
MIN = "min"
ACCURACY = "accuracy"
ACC = "acc"
RMSE = "rmse"
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
