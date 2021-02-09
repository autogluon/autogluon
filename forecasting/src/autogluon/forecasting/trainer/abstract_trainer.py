import time
import copy
from ..models.gluonts_model.abstract_gluonts.abstract_gluonts_model import AbstractGluonTSModel
import logging
from autogluon.core.utils.savers import save_pkl, save_json
from autogluon.core.utils.loaders import load_pkl
from collections import defaultdict
import pandas as pd
from gluonts.evaluation import Evaluator
from gluonts.evaluation.backtest import make_evaluation_predictions
import os

logger = logging.getLogger(__name__)

__all__ = ['AbstractTrainer']


class AbstractTrainer:
    trainer_file_name = 'trainer.pkl'
    trainer_info_name = 'info.pkl'
    trainer_info_json_name = 'info.json'

    def __init__(self, path: str, freq, prediction_length, scheduler_options=None, eval_metric=None,
                 save_data=True, **kwargs):
        self.path = path
        self.freq = freq
        self.prediction_length = prediction_length
        self.save_data = save_data
        self.quantiles = kwargs.get("quantiles", ["0.5"])
        self.is_data_saved = False
        self.model_full_dict = {}  # Dict of normal model -> FULL model. FULL models are produced by self.refit_single_full() and self.refit_ensemble_full().
        self._model_full_dict_val_score = {}  # Dict of FULL model -> normal model validation score in case the normal model had been deleted.
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

    @property
    def path_root(self) -> str:
        return self.path.rsplit(os.path.sep, maxsplit=2)[0] + os.path.sep

    @property
    def path_utils(self) -> str:
        return self.path_root + 'utils' + os.path.sep

    @property
    def path_data(self) -> str:
        return self.path_utils + 'data' + os.path.sep

    def save_train_data(self, data, verbose=True):
        path = self.path_data + 'train.pkl'
        save_pkl.save(path=path, object=data, verbose=verbose)

    def save_val_data(self, data, verbose=True):
        path = self.path_data + 'val.pkl'
        save_pkl.save(path=path, object=data, verbose=verbose)

    def load_train_data(self):
        path = self.path_data + 'train.pkl'
        return load_pkl.load(path=path)

    def load_val_data(self):
        path = self.path_data + 'val.pkl'
        return load_pkl.load(path=path)

    def load_data(self):
        train_data = self.load_train_data()
        val_data = self.load_val_data()
        return train_data, val_data

    def set_contexts(self, path_context):
        self.path, model_paths = self.create_contexts(path_context)
        for model, path in model_paths.items():
            self.set_model_attribute(model=model, attribute='path', val=path)

    def create_contexts(self, path_context: str) -> (str, dict):
        path = path_context
        model_paths = self.get_models_attribute_dict(attribute='path')
        for model, prev_path in model_paths.items():
            model_local_path = prev_path.split(self.path, 1)[1]
            new_path = path + model_local_path
            model_paths[model] = new_path

        return path, model_paths

    def get_models(self, hyperparameters):
        raise NotImplementedError

    def save(self):
        models = self.models
        if self.low_memory:
            self.models = {}
        try:
            save_pkl.save(path=self.path + self.trainer_file_name, object=self)
        except:
            self.models = {}
            save_pkl.save(path=self.path + self.trainer_file_name, object=self)
        if not self.models:
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
        model.save()

    def load_model(self, model_name, path=None, model_type=None) -> AbstractGluonTSModel:
        if isinstance(model_name, AbstractGluonTSModel):
            return model_name
        if model_name in self.models.keys():
            return self.models[model_name]
        else:
            if path is None:
                path = self.get_model_attribute(model=model_name, attribute='path')
            if model_type is None:
                model_type = self.get_model_attribute(model=model_name, attribute='type')
            return model_type.load(path=path, reset_path=self.reset_paths)

    def _add_model(self, model):
        self.model_info[model.name] = {}
        self.model_info[model.name]["path"] = model.path
        self.model_info[model.name]["type"] = type(model)
        self.model_info[model.name]["fit_time"] = model.fit_time
        self.model_info[model.name]["score"] = model.val_score

    def _train_single(self, train_data, model: AbstractGluonTSModel, time_limit=None):
        model.fit(train_data=train_data, time_limit=time_limit)
        return model

    def _train_single_full(self, train_data, model: AbstractGluonTSModel, val_data=None, hyperparameter_tune=False,
                           time_limit=None):
        if hyperparameter_tune:
            if self._scheduler_func is None or self._scheduler_options is None:
                raise ValueError('scheduler_options cannot be None when hyperparameter_tune = True')
            hpo_models, hpo_model_performances, hpo_results = model.hyperparameter_tune(train_data=train_data,
                                                                                        val_data=val_data,
                                                                                        scheduler_options=(
                                                                                            self._scheduler_func,
                                                                                            self._scheduler_options))
            self.hpo_results[model.name] = hpo_results
            model_names_trained = []
            for model_hpo_name, model_path in hpo_models.items():
                model_hpo = self.load_model(model_hpo_name, path=model_path, model_type=type(model))
                self._add_model(model_hpo)
                model_names_trained.append(model_hpo.name)
        else:
            model_names_trained = self._train_and_save(train_data, model=model, val_data=val_data,
                                                       time_limit=time_limit)

        return model_names_trained

    def _train_and_save(self, train_data, model: AbstractGluonTSModel, val_data=None, time_limit=None):
        fit_start_time = time.time()
        model_names_trained = []
        try:
            if time_limit is not None:
                if time_limit <= 0:
                    logging.log(15, f'Skipping {model.name} due to lack of time remaining.')
                    return model_names_trained
            else:
                logging.log(20, f'Fitting model: {model.name} ...')
            model = self._train_single(train_data, model)
            fit_end_time = time.time()
            if val_data is not None:
                score = -model.score(val_data)
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
            model.val_score = score
            self.save_model(model=model)
        except:
            pass
        else:
            self._add_model(model=model)
            model_names_trained.append(model.name)
            if self.low_memory:
                del model
        return model_names_trained

    def _train_multi(self, train_data, val_data=None, models=None, hyperparameters=None, hyperparameter_tune=False):
        if self.save_data and not self.is_data_saved:
            self.save_train_data(train_data)
            if val_data is not None:
                self.save_val_data(val_data)
            self.is_data_saved = True

        if hyperparameters is not None:
            hyperparameters = copy.deepcopy(hyperparameters)
        if models is None:
            models = self.get_models(hyperparameters)
        model_names_trained = []
        for i, model in enumerate(models):
            model_names_trained += self._train_single_full(train_data, model, val_data=val_data,
                                                           hyperparameter_tune=hyperparameter_tune)

        return model_names_trained

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

    def set_model_attribute(self, model, attribute: str, val):
        if not isinstance(model, str):
            model = model.name
        self.model_info[model][attribute] = val

    def leaderboard(self, data=None, extra_info=False):
        logger.log(30, "Generating leaderboard for all models trained...")
        model_names = self.get_model_names_all()
        score_val = []
        fit_time_marginal = []
        fit_order = list(range(1, len(model_names) + 1))
        score_dict = self.get_models_attribute_dict('score')
        fit_time_marginal_dict = self.get_models_attribute_dict('fit_time')
        for model_name in model_names:
            score_val.append(score_dict[model_name])
            fit_time_marginal.append(fit_time_marginal_dict[model_name])

        test_score = []
        if data is not None:
            logger.log(30, "Additional data provided, testing on the additional data...")
            for model_name in model_names:
                model = self.load_model(model_name)
                test_score.append(-model.score(data))
        df = pd.DataFrame(data={
            'model': model_names,
            'score': score_val,
            'fit_order': fit_order,
        })
        if test_score:
            df["test_score"] = test_score

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

    def predict(self, data, model=None, for_score=True, **kwargs):
        purpose = "evaluation" if for_score else "prediction"
        if model is None:
            logger.log(30, f"Does not specify model, "
                           f"will by default use the model with the best validation score for {purpose}")

        if model is not None:
            return self._predict_model(data, model, for_score, **kwargs)
        elif self.model_best is not None:
            return self._predict_model(data, self.model_best, for_score, **kwargs)
        else:
            model = self.get_model_best()
            self.model_best = model
            return self._predict_model(data, model, for_score, **kwargs)

    def score(self, data, model=None, quantiles=None):
        if self.eval_metric is not None and self.eval_metric not in ["MASE", "MAPE", "sMAPE", "mean_wQuantileLoss"]:
            raise ValueError(f"metric {self.eval_metric} is not available yet.")

        # if quantiles are given, use the given on, otherwise use the default
        if quantiles is not None:
            evaluator = Evaluator(quantiles=quantiles)
        else:
            evaluator = Evaluator(quantiles=self.quantiles)
        forecasts, tss = self.predict(data, model=model, for_score=True)
        num_series = len(tss)
        agg_metrics, item_metrics = evaluator(iter(tss), iter(forecasts), num_series=num_series)
        return agg_metrics[self.eval_metric]

    def _predict_model(self, data, model, for_score=True, **kwargs):
        if isinstance(model, str):
            model = self.load_model(model)
        if for_score:
            forecasts, tss = model.predict_for_scoring(data, **kwargs)
            return forecasts, tss
        else:
            return model.predict(data, **kwargs)

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

    def refit_single_full(self, train_data=None, val_data=None, models=None):
        models_trained_full = []
        model_full_dict = {}
        # assume val data will contain all train data.
        if train_data is None:
            train_data = self.load_val_data()

        if models is None:
            self.get_model_names_all()

        for model in models:
            model = self.load_model(model)
            model_name = model.name
            model_full = model.convert_to_refit_full_template()
            models_trained = self._train_multi(train_data=train_data, val_data=None, hyperparameters=None,
                                               hyperparameter_tune=False, models=[model_full])

            if len(models_trained) == 1:
                model_full_dict[model_name] = models_trained[0]
            for model_trained in models_trained:
                self._model_full_dict_val_score[model_trained] = self.get_model_attribute(model_name, 'score')
            models_trained_full += models_trained

        self.model_full_dict.update(model_full_dict)
        self.save()
        return models_trained_full

    def refit_full(self, models='all'):
        if isinstance(models, str):
            if models == 'all':
                models = self.get_model_names_all()
            elif models == 'best':
                models = [self.get_model_best()]
            else:
                models = self.load_model(models)
        existing_models = self.get_model_names_all()
        valid_model_set = []
        for model in models:
            if model in self.model_full_dict and self.model_full_dict[model] in existing_models:
                logger.log(20,
                           f"Model '{model}' already has a refit _FULL model: '{self.model_full_dict[model]}', skipping refit...")
            else:
                valid_model_set.append(model)

        if valid_model_set:
            models_trained_full = self.refit_single_full(models=valid_model_set)
        else:
            models_trained_full = []

        for model_full in models_trained_full:
            # TODO: leave space for future ensemble?
            pass

        self.save()
        return copy.deepcopy(self.model_full_dict)
