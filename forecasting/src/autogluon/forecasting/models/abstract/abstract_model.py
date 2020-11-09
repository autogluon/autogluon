from gluonts.evaluation.backtest import make_evaluation_predictions
import core.utils.savers.save_pkl as save_pkl
import core.utils.loaders.load_pkl as load_pkl
import os
from gluonts.evaluation import Evaluator
from gluonts.evaluation.backtest import make_evaluation_predictions
from tqdm import tqdm


class AbstractModel:

    def __init__(self):
        pass


