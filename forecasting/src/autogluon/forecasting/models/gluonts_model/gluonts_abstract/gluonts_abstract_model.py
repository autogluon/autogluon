from gluonts.evaluation.backtest import make_evaluation_predictions
import core.utils.savers.save_pkl as save_pkl
import core.utils.loaders.load_pkl as load_pkl
import os
from ...abstract.abstract_model import AbstractModel
from gluonts.evaluation import Evaluator
from gluonts.evaluation.backtest import make_evaluation_predictions
from tqdm import tqdm


class GluonTSAbstractModel(AbstractModel):

    model_file_name = "model.pkl"

    def __init__(self, path: str, freq: str, prediction_length: int, name: str, eval_metric: str = None,
                 hyperparameters=None, model=None):
        """
        Create a new model
        Args:
            path(str): directory where to store all the model
            name(str): name of subdirectory inside path where model will be saved.
            eval_metric(str): objective function the model intends to optimize, will use mean_wQuantileLoss by default
            hyperparameters: various hyperparameters that will be used by model (can be search spaces instead of fixed values).
        """
        super().__init__()
        self.name = name
        self.path_root = path
        self.path_suffix = self.name + os.path.sep
        self.path = self.path_root + self.path_suffix

        if eval_metric is None:
            eval_metric = "mean_wQuantileLoss"
        if eval_metric is not None and eval_metric not in ["MASE", "MAPE", "sMAPE", "mean_wQuantileLoss"]:
            raise ValueError(f"metric {eval_metric} is not available yet.")
        self.eval_metric = eval_metric

        self.params = {}
        self.set_default_parameters()
        self.params["freq"] = freq
        self.params["prediction_length"] = prediction_length
        self.nondefault_parameters = []
        if hyperparameters is not None:
            self.params.update(hyperparameters)
            self.nondefault_parameters = list(hyperparameters.keys())[:]

        self.model = model

    def set_contexts(self, path_context):
        self.path = self.create_contexts(path_context)
        self.path_suffix = self.name + os.path.sep
        self.path_root = self.path.rsplit(self.path_suffix, 1)[0]

    @staticmethod
    def create_contexts(path_context):
        path = path_context
        return path

    def save(self, path: str = None,):
        if path is None:
            path = self.path
        file_path = path + self.model_file_name
        save_pkl.save(path=file_path, object=self)
        return path

    @classmethod
    def load(cls, path: str, reset_path=True):
        file_path = path + cls.model_file_name
        model = load_pkl.load(path=file_path,)
        if reset_path:
            model.set_context(path)
        return model

    def set_default_parameters(self):
        pass

    def create_model(self):
        pass

    def fit(self, train_data):
        pass
        # self.model = self.model.train(train_data)

    def predict(self, test_data, num_samples=100):
        forecast_it, ts_it = make_evaluation_predictions(dataset=test_data,
                                                         predictor=self.model,
                                                         num_samples=num_samples)
        return list(tqdm(forecast_it, total=len(test_data))), list(tqdm(ts_it, total=len(test_data)))

    def hyperparameter_tune(self, train_data, test_data, metric, scheduler_options, **kwargs):
        pass

    def score(self, y, metric=None):
        """
        metric: if metric is None, we will by default use mean_wQuantileLoss for scoring.
                should be one of "MASE", "MAPE", "sMAPE", "mean_wQuantileLoss"
        """
        if metric is None:
            metric = self.eval_metric
        if metric is not None and metric not in ["MASE", "MAPE", "sMAPE", "mean_wQuantileLoss"]:
            raise ValueError(f"metric {metric} is not available yet.")

        # if quantiles are given, use the given on, otherwise use the default
        if "quantiles" in self.params:
            evaluator = Evaluator(quantiles=self.params["quantiles"])
        else:
            evaluator = Evaluator()

        forecasts, tss = self.predict(y)
        num_series = len(tss)
        agg_metrics, item_metrics = evaluator(iter(tss), iter(forecasts), num_series=num_series)
        return agg_metrics[metric]