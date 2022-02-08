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
val_metric_name = {BINARY: VAL_ACC, MULTICLASS: VAL_ACC}
# minmax_mode = {VAL_ACC: MAX, VAL_LOSS: MIN}
minmax_mode = {ACCURACY: MAX, ACC: MAX, RMSE: MIN, QUADRATIC_KAPPA: MAX, VAL_ACC: MAX, VAL_LOSS: MIN}
test_metric_name = {BINARY: TEST_ACC, MULTICLASS: TEST_ACC}

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
# CONFIG_DIR = "config_dir"
MODEL = "model"
DATA = "data"
OPTIMIZATION = "optimization"
ENVIRONMENT = "environment"



