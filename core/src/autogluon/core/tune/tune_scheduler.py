import logging
from collections import OrderedDict
from functools import partial

from ray import tune
from ray.tune import Callback

from autogluon.core.space import *
from autogluon.core.utils.edict import EasyDict

logger = logging.getLogger(__name__)


class ResultsHistoryCallback(Callback):

    def __init__(self, collector: dict):
        self.collector = collector

    def on_trial_result(self, iteration, trials, trial, result, **info):
        result['time_this_iter'] = result['time_this_iter_s']
        result['time_since_start'] = result['time_total_s']
        if trial in self.collector:
            self.collector[trial].append(result)
        else:
            self.collector[trial] = [result]


class RayTuneScheduler(object):

    def __init__(self, task_fn, reward_attr, time_attr='epoch', **tune_args):
        self.task_fn = task_fn
        self.tune_args = tune_args
        self.tune_args['metric'] = reward_attr
        self.training_history = None
        self.reward_attr = reward_attr
        self.time_attr = time_attr

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

    @classmethod
    def reporter_fn(cls, **kwargs):
        tune.report(**kwargs)

    @classmethod
    def train_fn_wrapper(cls, fn, config, reporter=None, **kwargs):
        config = EasyDict(config)
        return fn(config, reporter=cls.reporter_fn, **kwargs)

    def run(self, **kwargs):
        self.training_history = OrderedDict()

        results_history_callback = ResultsHistoryCallback(self.training_history)
        result = tune.run(
            partial(self.train_fn_wrapper, self.task_fn.f),
            name='AG',
            callbacks=[results_history_callback],
            config=self.wrap_space(self.task_fn.kwvars),
            **self.tune_args
        )
        best_trial = result.get_best_trial("accuracy", "max", "last")
        print("Best trial config: {}".format(best_trial.config))
        print("Best trial final validation accuracy: {}".format(best_trial.last_result["accuracy"]))

        return result

    def restore(self, **kwargs):
        results_history_callback = ResultsHistoryCallback(self.training_history)
        result = tune.run(
            partial(self.train_fn_wrapper, self.task_fn.f),
            name='AG',
            callbacks=[results_history_callback],
            config=self.wrap_space(self.task_fn.kwvars),
            **kwargs
        )
        best_trial = result.get_best_trial("accuracy", "max", "last")
        print("Best trial config: {}".format(best_trial.config))
        print("Best trial final validation accuracy: {}".format(best_trial.last_result["accuracy"]))

        return result


    def join_jobs(self, timeout=None):
        # Keep for compatibility
        pass

    def get_training_curves(self, filename=None, plot=False, use_legend=True):
        """Get Training Curves

        Parameters
        ----------
            filename : str
            plot : bool
            use_legend : bool

        Examples
        --------
        >>> scheduler.run()
        >>> scheduler.join_jobs()
        >>> scheduler.get_training_curves(plot=True)

            .. image:: https://github.com/zhanghang1989/AutoGluonWebdata/blob/master/doc/api/autogluon.1.png?raw=true
        """
        if filename is None and not plot:
            logger.warning('Please either provide filename or allow plot in get_training_curves')
        import matplotlib.pyplot as plt
        plt.ylabel(self.reward_attr)
        plt.xlabel(self.time_attr)
        plt.title("Performance vs Training-Time in each HPO Trial")
        for task_id, task_res in self.training_history.items():
            rewards = [x[self.reward_attr] for x in task_res]
            x = [x[self.time_attr] for x in task_res]
            plt.plot(x, rewards, label=f'task {task_id}')
        if use_legend:
            plt.legend(loc='best')
        if filename:
            logger.info(f'Saving Training Curve in {filename}')
            plt.savefig(filename)
        if plot:
            plt.show()
