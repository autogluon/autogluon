from autogluon.core.models.abstract.abstract_model import AbstractModel

from .lgb.lgb_model import LGBModel
from .catboost.catboost_model import CatBoostModel
from .xgboost.xgboost_model import XGBoostModel
from .rf.rf_model import RFModel
from .xt.xt_model import XTModel
from .knn.knn_model import KNNModel
from .lr.lr_model import LinearModel
from .tabular_nn.mxnet.tabular_nn_mxnet import TabularNeuralNetMxnetModel
from .tabular_nn.torch.tabular_nn_torch import TabularNeuralNetTorchModel
from .fastainn.tabular_nn_fastai import NNFastAiTabularModel
from .fasttext.fasttext_model import FastTextModel
from .text_prediction.text_prediction_v1_model import TextPredictorModel
from .image_prediction.image_predictor import ImagePredictorModel
from .imodels.imodels_models import RuleFitModel, BoostedRulesModel, GreedyTreeModel, HSTreeModel, \
    FigsModel, _IModelsModel
from .vowpalwabbit.vowpalwabbit_model import VowpalWabbitModel
from .automm.automm_model import MultiModalPredictorModel
from .automm.ft_transformer import FTTransformerModel
