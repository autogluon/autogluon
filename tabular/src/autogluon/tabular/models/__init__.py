from autogluon.core.models.abstract.abstract_model import AbstractModel

from .lgb.lgb_model import LGBModel
from .catboost.catboost_model import CatBoostModel
from .xgboost.xgboost_model import XGBoostModel
from .rf.rf_model import RFModel
from .xt.xt_model import XTModel
from .knn.knn_model import KNNModel
from .lr.lr_model import LinearModel
from .tabular_nn.tabular_nn_model import TabularNeuralNetModel
from .tabular_nn.tabular_nn_quantile import TabularNeuralQuantileModel
from .fastainn.tabular_nn_fastai import NNFastAiTabularModel
from .fasttext.fasttext_model import FastTextModel
from .text_prediction.text_prediction_v1_model import TextPredictorModel
from .image_prediction.image_predictor import ImagePredictorModel
