from enum import Enum


class Task(str, Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"


class FeatureType(str, Enum):
    NUMERICAL = "numerical"
    CATEGORICAL = "categorical"
    MIXED = "mixed"


class SearchType(str, Enum):
    DEFAULT = "default"
    RANDOM = "random"


class DatasetSize(int, Enum):
    SMALL = 1000
    MEDIUM = 10000
    LARGE = 50000


class DataSplit(str, Enum):
    TRAIN = "train"
    VALID = "valid"
    TEST = "test"


class Phase(str, Enum):
    TRAINING = "training"
    VALIDATION = "validation"
    TESTING = "testing"


class ModelName(str, Enum):
    PLACEHOLDER = "_placeholder_"   # This is a placeholder for the current running model
    FT_TRANSFORMER = "FT-Transformer"
    TABPFN = "TabPFN"
    FOUNDATION = "Foundation"
    FOUNDATION_FLASH = "FoundationFlash"
    TAB2D = "Tab2D"
    TAB2D_COL_ROW = "Tab2D_COL_ROW"
    TAB2D_SDPA = "Tab2D_SDPA"
    SAINT = "SAINT"
    MLP = "MLP"
    MLP_RTDL = "MLP-rtdl"
    RESNET = "Resnet"
    RANDOM_FOREST = "RandomForest"
    XGBOOST = "XGBoost"
    CATBOOST = "CatBoost"
    LIGHTGBM = "LightGBM"
    GRADIENT_BOOSTING_TREE = "GradientBoostingTree"
    HIST_GRADIENT_BOOSTING_TREE = "HistGradientBoostingTree"
    LOGISTIC_REGRESSION = "LogisticRegression"
    LINEAR_REGRESSION = "LinearRegression"
    DECISION_TREE = "DecisionTree"
    KNN = "KNN"
    STG = "STG"
    SVM = "SVM"
    TABNET = "TabNet"
    TABTRANSFORMER = "TabTransformer"
    DEEPFM = "DeepFM"
    VIME = "VIME"
    DANET = "DANet"
    NODE = "NODE"
    AUTOGLUON = "AutoGluon"


class ModelClass(str, Enum):
    BASE = 'base'
    GBDT = 'GBDT'
    NN = 'NN'
    ICLT = 'ICLT'


class DownstreamTask(str, Enum):
    ZEROSHOT = "zeroshot"
    FINETUNE = "finetune"



class BenchmarkName(str, Enum):
    DEBUG_CLASSIFICATION = "debug_classification"
    DEBUG_REGRESSION = "debug_regression"
    DEBUG_TABZILLA = "debug_tabzilla"

    CATEGORICAL_CLASSIFICATION = "categorical_classification"
    NUMERICAL_CLASSIFICATION = "numerical_classification"
    CATEGORICAL_REGRESSION = "categorical_regression"
    NUMERICAL_REGRESSION = "numerical_regression"
    CATEGORICAL_CLASSIFICATION_LARGE = "categorical_classification_large"
    NUMERICAL_CLASSIFICATION_LARGE = "numerical_classification_large"
    CATEGORICAL_REGRESSION_LARGE = "categorical_regression_large"
    NUMERICAL_REGRESSION_LARGE = "numerical_regression_large"
    
    TABZILLA_HARD = "tabzilla_hard"
    TABZILLA_HARD_MAX_TEN_CLASSES = "tabzilla_hard_max_ten_classes"
    TABZILLA_HAS_COMPLETED_RUNS = "tabzilla_has_completed_runs"


class BenchmarkOrigin(str, Enum):
    TABZILLA = "tabzilla"
    WHYTREES = "whytrees"


class GeneratorName(str, Enum):
    TABPFN = 'tabpfn'
    TREE = 'tree'
    RANDOMFOREST = 'randomforest'
    NEIGHBOR = 'neighbor'
    MIX = 'mix'
    PERLIN = 'perlin'
    MIX_7 = 'mix_7'
    MIX_6 = 'mix_6'
    MIX_5 = 'mix_5'
    MIX_5_GP = 'mix_5_gp'
    MIX_4 = 'mix_4'
    MIX_4_AG = 'mix_4_ag'
    LR = 'lr'
    POLY = 'poly'
    SAMPLE_RF = 'sample_rf'
    SAMPLE_GP = 'sample_gp'
    TABREPO = 'tabrepo'
    MIX_4_TABREPO = 'mix_4_tabrepo'
    MIX_4_TABPFNV2 = 'mix_4_tabpfnv2'


class MetricName(str, Enum):
    ACCURACY = "accuracy"
    F1 = "f1"
    AUC = "auc"
    MSE = "mse"
    MAE = "mae"
    R2 = "r2"
    LOG_LOSS = "log_loss"
    RMSE = "rmse"


class LossName(str, Enum):
    CROSS_ENTROPY = "cross_entropy"
    MSE = "mse"
    MAE = "mae"