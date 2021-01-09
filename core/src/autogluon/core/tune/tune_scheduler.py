import logging
from collections import OrderedDict
from functools import partial

from ray import tune
from ray.tune import Callback

from autogluon.core.space import *
from autogluon.core.utils.edict import EasyDict

logger = logging.getLogger(__name__)


class ResultsHistoryCallback(Callback):

    def __init__(self, training_history: dict, config_history: dict):
        self.training_history = training_history
        self.config_history = config_history

    def on_trial_result(self, iteration, trials, trial, result, **info):
        result['time_this_iter'] = result['time_this_iter_s']
        result['time_since_start'] = result['time_total_s']
        trial = str(trial)
        if trial in self.training_history:
            self.training_history[trial].append(result)
            self.config_history[trial].append(result['config'])
        else:
            self.training_history[trial] = [result]
            self.config_history[trial] = [result['config']]


class TuneReporter:
    def __call__(self, *args, **kwargs):
        tune.report(**kwargs)

    def terminate(self):
        pass  # compatibility


class RayTuneScheduler(object):

    def __init__(self, task_fn, reward_attr, time_attr='epoch', **tune_args):
        self.task_fn = task_fn
        self.tune_args = tune_args['tune_args'] if 'tune_args' in tune_args else tune_args
        self.tune_args['metric'] = reward_attr
        self.training_history = None
        self.config_history = None
        self._reward_attr = reward_attr
        self.time_attr = time_attr
        self.result = None
        self.resource = self.tune_args['resources_per_trial']

        self.metadata = {
            'search_space': task_fn.kwspaces,
            'search_strategy': 'N/A',
            'stop_criterion': {
                'time_limits': self.tune_args.get('time_budget_s', None),
                # 'max_reward': self.max_reward
            },
            'resources_per_trial': self.resource}

    def train_fn_wrapper(self, fn, config, reporter=None, **kwargs):
        config = EasyDict(config)
        config['task_id'] = tune.session.get_session().trial_name
        return fn(config, reporter=TuneReporter(), **kwargs)

    def run(self, **kwargs):
        self.training_history = OrderedDict()
        self.config_history = OrderedDict()
        self.result = None

        results_history_callback = ResultsHistoryCallback(self.training_history, self.config_history)
        result = tune.run(
            partial(self.train_fn_wrapper, self.task_fn.f),
            name='AG',
            callbacks=[results_history_callback],
            config=self.wrap_space(self.task_fn.kwvars),
            **self.tune_args
        )
        self.result = result

    def restore(self, **kwargs):
        results_history_callback = ResultsHistoryCallback(self.training_history, self.config_history)
        result = tune.run(
            partial(self.train_fn_wrapper, self.task_fn.f),
            name='AG',
            callbacks=[results_history_callback],
            config=self.wrap_space(self.task_fn.kwvars),
            **kwargs
        )
        self.result = result

    @classmethod
    def wrap_space(cls, space):
        if isinstance(space, Real):
            fn = tune.loguniform if space.log else tune.uniform
            return fn(space.lower, space.upper)
        elif isinstance(space, Int):
            return tune.randint(space.lower, space.upper + 1)
        elif isinstance(space, Categorical):
            return tune.choice(space)
        elif type(space) in [List, list]:
            return [cls.wrap_space(s) for s in space]
        elif type(space) in [Dict, dict]:
            return {k: cls.wrap_space(v) for k, v in space.items()}
        else:
            return space

    def get_best_config(self):
        best_trial = self.result.get_best_trial(self._reward_attr, self.tune_args['mode'], "last")
        return best_trial.config

    def get_best_reward(self):
        best_trial = self.result.get_best_trial(self._reward_attr, self.tune_args['mode'], "last")
        return best_trial.last_result[self._reward_attr]

    def join_jobs(self, timeout=None):
        # Keep for compatibility
        pass

    def get_training_curves(self, filename=None, plot=False, use_legend=True):
        if filename is None and not plot:
            logger.warning('Please either provide filename or allow plot in get_training_curves')
        import matplotlib.pyplot as plt
        plt.ylabel(self._reward_attr)
        plt.xlabel(self.time_attr)
        plt.title("Performance vs Training-Time in each HPO Trial")
        for task_id, task_res in self.training_history.items():
            rewards = [x[self._reward_attr] for x in task_res]
            x = [x[self.time_attr] for x in task_res]
            plt.plot(x, rewards, label=f'task {task_id}')
        if use_legend:
            plt.legend(loc='best')
        if filename:
            logger.info(f'Saving Training Curve in {filename}')
            plt.savefig(filename)
        if plot:
            plt.show()
