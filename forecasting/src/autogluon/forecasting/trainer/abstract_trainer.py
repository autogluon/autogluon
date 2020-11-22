import time
import copy
from ..models.gluonts_model.gluonts_abstract.gluonts_abstract_model import GluonTSAbstractModel
import networkx as nx
import logging
from core.utils.savers import save_pkl, save_json
from core.utils.loaders import load_pkl
from collections import defaultdict
import pandas as pd
from gluonts.evaluation import Evaluator
from gluonts.evaluation.backtest import make_evaluation_predictions


logger = logging.getLogger(__name__)

__all__ = ['AbstractTrainer']


class AbstractTrainer:

    trainer_file_name = 'trainer.pkl'
    trainer_info_name = 'info.pkl'
    trainer_info_json_name = 'info.json'

    def __init__(self, path: str, freq, prediction_length, scheduler_options=None, eval_metric=None,
                 save_data=False):
        self.path = path
        self.freq = freq
        self.prediction_length = prediction_length
        self.save_data = save_data

        # do we really need the graph here as what is done in tabular?
        self.model_graph = nx.DiGraph()
        self.model_info = {}
        self.models = {}
        self.model_best = None

        self.reset_paths = False

        if eval_metric is not None:
            self.eval_metric = eval_metric
        else:
            self.eval_metric = "mean_wQuantileLoss"

        if scheduler_options is not None:
            self._scheduler_func = scheduler_options[0]  # unpack tuple
            self._scheduler_options = scheduler_options[1]
        else:
            self._scheduler_func = None
            self._scheduler_options = None

        self.hpo_results = {}

        self.low_memory = False

        self.hyperparameters = {}

    def get_models(self, hyperparameters):
        raise NotImplementedError

    def save(self):
        models = self.models
        if self.low_memory:
            self.models = {}
        save_pkl.save(path=self.path + self.trainer_file_name, object=self)
        if self.low_memory:
            self.models = models

    @classmethod
    def load(cls, path, reset_paths=False):
        load_path = path + cls.trainer_file_name
        if not reset_paths:
            return load_pkl.load(path=load_path)
        else:
            obj = load_pkl.load(path=load_path)
            obj.set_contexts(path)
            obj.reset_paths = reset_paths
            return obj

    def save_model(self, model):
        if self.low_memory:
            model.save()
        else:
            self.models[model.name] = model

    def load_model(self, model_name, path=None, model_type=None) -> GluonTSAbstractModel:
        if isinstance(model_name, GluonTSAbstractModel):
            return model_name
        if model_name in self.models.keys():
            return self.models[model_name]
        else:
            # TODO: get_model_attribute()
            # if path is None:
            #     path = self.get_model_attribute(model=model_name, attribute='path')
            # if model_type is None:
            #     model_type = self.get_model_attribute(model=model_name, attribute='type')
            return model_type.load(path=path, reset_path=self.reset_paths)

    def _add_model(self, model):
        print(model.name, model.path, type(model))
        # self.model_graph.add_node(
        #     model.name,
        #     path=model.path,
        #     type=type(model),
        # )
        self.model_info[model.name] = {}
        self.model_info[model.name]["path"] = model.path
        self.model_info[model.name]["type"] = type(model)
        self.model_info[model.name]["fit_time"] = model.fit_time
        self.model_info[model.name]["score"] = model.test_score

    def _train_single(self, train_data, model: GluonTSAbstractModel, time_limit=None):
        model.fit(train_data=train_data)
        return model

    def _train_single_full(self, train_data, model: GluonTSAbstractModel, test_data=None, hyperparameter_tune=False, time_limit=None):
        if hyperparameter_tune:
            if self._scheduler_func is None or self._scheduler_options is None:
                raise ValueError('scheduler_options cannot be None when hyperparameter_tune = True')
            hpo_models, hpo_model_performances, hpo_results = model.hyperparameter_tune(train_data=train_data,
                                                                                        test_data=test_data,
                                                                                        scheduler_options=(self._scheduler_func, self._scheduler_options))
            self.hpo_results[model.name] = hpo_results
            model_names_trained = []
            for model_hpo_name, model_path in hpo_models.items():
                model_hpo = self.load_model(model_hpo_name, path=model_path, model_type=type(model))
                self._add_model(model_hpo)
                model_names_trained.append(model_hpo.name)
            # self.model_best = self.get_model_best()
        else:
            model_names_trained = self._train_and_save(train_data, model=model, test_data=test_data, time_limit=time_limit)

        return model_names_trained

    def _train_and_save(self, train_data, model: GluonTSAbstractModel, test_data=None, time_limit=None):
        fit_start_time = time.time()
        model_names_trained = []
        try:
            if time_limit is not None:
                if time_limit <= 0:
                    logging.log(15, f'Skipping {model.name} due to lack of time remaining.')
                    return model_names_trained
                logging.log(20, f'Fitting model: {model.name} ... Training model for up to {round(time_limit, 2)}s of the {round(time_left_total, 2)}s of remaining time.')
            else:
                logging.log(20, f'Fitting model: {model.name} ...')
            model = self._train_single(train_data, model)
            fit_end_time = time.time()
            if test_data is not None:
                score = model.score(test_data)
            else:
                score = None
            pred_end_time = time.time()
            if model.fit_time is None:
                model.fit_time = fit_end_time - fit_start_time
            if model.predict_time is None:
                if score is None:
                    model.predict_time = None
                else:
                    model.predict_time = pred_end_time - fit_end_time
            model.test_score = score
            self.save_model(model=model)
        except:
            pass
        else:
            self._add_model(model=model)
            model_names_trained.append(model.name)
            if self.low_memory:
                del model
        return model_names_trained

    def _train_multi(self, train_data, test_data=None, hyperparameters=None, hyperparameter_tune=False):
        if hyperparameters is None:
            hyperparameters = self.hyperparameters
        hyperparameters = copy.deepcopy(hyperparameters)
        models = self.get_models(hyperparameters)

        for i, model in enumerate(models):
            self._train_single_full(train_data, model, test_data=test_data, hyperparameter_tune=hyperparameter_tune)

    def get_model_names_all(self):
        return list(self.model_info.keys())

    def get_models_attribute_dict(self, attribute, models=None):
        results = {}
        if models is None:
            models = self.get_model_names_all()
        for model in models:
            results[model] = self.model_info[model][attribute]
        return results

    def get_model_best(self):
        models = self.get_model_names_all()
        if not models:
            raise AssertionError('Trainer has no fit models that can infer.')
        model_performances = self.get_models_attribute_dict(attribute='score')
        perfs = [(m, model_performances[m]) for m in models if model_performances[m] is not None]
        return max(perfs, key=lambda i: i[1])[0]

    def get_model_attribute(self, model, attribute: str):
        if not isinstance(model, str):
            model = model.name
        return self.model_info[model][attribute]

    def leaderboard(self, extra_info=False):
        model_names = self.get_model_names_all()
        score_val = []
        fit_time_marginal = []
        fit_order = list(range(1,len(model_names)+1))
        score_dict = self.get_models_attribute_dict('score')
        fit_time_marginal_dict = self.get_models_attribute_dict('fit_time')
        for model_name in model_names:
            score_val.append(score_dict[model_name])
            fit_time_marginal.append(fit_time_marginal_dict[model_name])

        df = pd.DataFrame(data={
            'model': model_names,
            'score': score_val,
            'fit_order': fit_order,
        })

        df_sorted = df.sort_values(by=['score', 'model'], ascending=[False, False]).reset_index(drop=True)

        df_columns_lst = df_sorted.columns.tolist()
        explicit_order = [
            'model',
            'score',
            'fit_order'
        ]
        explicit_order = [column for column in explicit_order if column in df_columns_lst]
        df_columns_other = [column for column in df_columns_lst if column not in explicit_order]
        df_columns_new = explicit_order + df_columns_other
        df_sorted = df_sorted[df_columns_new]

        return df_sorted

    def predict(self, data, model=None):
        if model is not None:
            return self._predict_model(data, model)
        elif self.model_best is not None:
            return self._predict_model(data, self.model_best)
        else:
            model = self.get_model_best()
            self.model_best = model
            return self._predict_model(data, model)

    def score(self, data, model=None, quantiles=None):
        if self.eval_metric is not None and self.eval_metric not in ["MASE", "MAPE", "sMAPE", "mean_wQuantileLoss"]:
            raise ValueError(f"metric { self.eval_metric} is not available yet.")

        # if quantiles are given, use the given on, otherwise use the default
        if quantiles is not None:
            evaluator = Evaluator(quantiles=quantiles)
        else:
            evaluator = Evaluator()

        forecasts, tss = self.predict(data, model=model)
        num_series = len(tss)
        agg_metrics, item_metrics = evaluator(iter(tss), iter(forecasts), num_series=num_series)
        return agg_metrics[self.eval_metric]

    def _predict_model(self, data, model):
        if isinstance(model, str):
            model = self.load_model(model)
        forecasts, tss = model.predict(data)
        return forecasts, tss

    @classmethod
    def load_info(cls, path, reset_paths=False, load_model_if_required=True):
        load_path = path + cls.trainer_info_name
        try:
            return load_pkl.load(path=load_path)
        except:
            if load_model_if_required:
                trainer = cls.load(path=path, reset_paths=reset_paths)
                return trainer.get_info()
            else:
                raise

    def save_info(self, include_model_info=False):
        info = self.get_info(include_model_info=include_model_info)

        save_pkl.save(path=self.path + self.trainer_info_name, object=info)
        save_json.save(path=self.path + self.trainer_info_json_name, obj=info)
        return info

    def get_info(self, include_model_info=False):
        num_models_trained = len(self.get_model_names_all())
        if self.model_best is not None:
            best_model = self.model_best
        else:
            try:
                best_model = self.get_model_best()
            except AssertionError:
                best_model = None
        if best_model is not None:
            best_model_score_val = self.get_model_attribute(model=best_model, attribute='score')
        else:
            best_model_score_val = None

        info = {
            'best_model': best_model,
            'best_model_score_val': best_model_score_val,
            'num_models_trained': num_models_trained,
        }

        if include_model_info:
            info['model_info'] = self.get_models_info()

        return info

    def get_models_info(self, models=None):
        if models is None:
            models = self.get_model_names_all()
        model_info_dict = dict()
        for model in models:
            if isinstance(model, str):
                if model in self.models.keys():
                    model = self.models[model]
            if isinstance(model, str):
                model_type = self.get_model_attribute(model=model, attribute='type')
                model_path = self.get_model_attribute(model=model, attribute='path')
                model_info_dict[model] = model_type.load_info(path=model_path)
            else:
                model_info_dict[model.name] = model.get_info()
        return model_info_dict

