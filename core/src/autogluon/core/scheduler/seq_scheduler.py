import logging
import pickle
import time
from collections import OrderedDict
from copy import deepcopy

from autogluon.core.searcher import GPFIFOSearcher
from autogluon.core.utils import EasyDict
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


class LocalReporter:
    def __init__(self, trial, config, training_history: dict, config_history: dict):
        self.trial = trial

        self.training_history = training_history
        if trial not in self.training_history:
            self.training_history[trial] = []

        self.task_config = EasyDict(deepcopy(config))

        self.config_history = config_history

        self.trial_started = time.time()
        self.last_reported_time = self.trial_started
        self.last_result = None

    def __call__(self, *args, **kwargs):
        result = deepcopy(kwargs)

        if 'done' not in result:
            result['trial'] = self.trial

            now = time.time()
            result['time_this_iter'] = now - self.last_reported_time
            result['time_since_start'] = now - self.trial_started
            self.last_reported_time = now

            self.training_history[self.trial].append(result)

            if self.trial not in self.config_history:
                self.config_history[self.trial] = self.task_config
                if 'util_args' in self.task_config:
                    self.task_config.pop('util_args')

            self.last_result = result
        if 'traceback' in result:
            self.searcher.evaluation_failed(config=self.task_config, **result)

    def terminate(self):
        pass  # compatibility


class LocalSequentialScheduler(object):

    def __init__(self, train_fn, reward_attr, time_attr='epoch', searcher='auto', **kwargs):
        self.train_fn = train_fn
        self.training_history = None
        self.config_history = None
        self._reward_attr = reward_attr
        self.time_attr = time_attr
        self.resource = kwargs['resource']
        self.max_reward = kwargs.get('max_reward')

        if kwargs.get('num_trials') is None:
            assert kwargs.get('time_out') is not None, "Need stopping criterion: Either num_trials or time_out"

        if searcher is 'local_sequential_auto':
            searcher = 'auto'

        if searcher is 'auto':
            searcher = GPFIFOSearcher
        self.searcher = searcher(self.train_fn.cs, reward_attribute=self._reward_attr)

        self.num_trials = kwargs.get('num_trials', 9999)
        self.time_out = kwargs.get('time_out')
        if self.num_trials is None:
            assert self.time_out is not None, \
                "Need stopping criterion: Either num_trials or time_out"

        self.metadata = {
            'search_space': train_fn.kwspaces,
            'search_strategy': searcher,
            'stop_criterion': {
                'time_limits': self.time_out,
                'max_reward': self.max_reward},
            'resources_per_trial': self.resource}

    def run(self, **kwargs):
        self.searcher.configure_scheduler(self)

        self.training_history = OrderedDict()
        self.config_history = OrderedDict()

        time_start = time.time()

        avg_trial_run_time = None
        for i in tqdm(range(self.num_trials)):
            trial_start_time = time.time()
            self.run_trial(task_id=i)
            trial_end_time = time.time()

            if self.max_reward and self.get_best_reward() >= self.max_reward:
                logger.log(20, f'\tMax reward is reached')
                break

            if self.time_out is not None:
                avg_trial_run_time = self.get_average_trial_time_(i, avg_trial_run_time, trial_start_time, trial_end_time)
                if not self.has_enough_time_for_trial_(self.time_out, time_start, trial_start_time, trial_end_time, avg_trial_run_time):
                    logger.log(20, f'\tTime limit exceeded...')
                    break

    @classmethod
    def has_enough_time_for_trial_(cls, time_out, time_start, trial_start_time, trial_end_time, avg_trial_run_time, fill_factor=0.9):
        time_spent = trial_end_time - time_start
        is_timeout_exceeded = time_spent >= time_out
        time_left = time_start + time_out - trial_end_time
        is_enough_time_for_another_trial = True
        if avg_trial_run_time:
            is_enough_time_for_another_trial = time_left > avg_trial_run_time * fill_factor
        return is_enough_time_for_another_trial and not is_timeout_exceeded

    @classmethod
    def get_average_trial_time_(cls, i, avg_trial_run_time, trial_start_time, time_end):
        trial_time = time_end - trial_start_time
        if avg_trial_run_time is None:
            avg_trial_run_time = trial_time
        else:
            avg_trial_run_time = ((avg_trial_run_time * i) + trial_time) / (i + 1)
        return avg_trial_run_time

    def run_trial(self, task_id=0):
        searcher_config = self.searcher.get_config()
        reporter = LocalReporter(task_id, searcher_config, self.training_history, self.config_history)
        task_config = deepcopy(EasyDict(self.train_fn.kwvars))
        task_config['task_id'] = task_id
        self.searcher.register_pending(searcher_config)
        self.train_fn(task_config, config=searcher_config, reporter=reporter)
        if reporter.last_result:
            self.searcher.update(config=searcher_config, **reporter.last_result)

    def join_jobs(self, timeout=None):
        # Required by autogluon
        pass

    def get_best_config(self):
        return self.searcher.get_best_config()

    def get_best_reward(self):
        return self.searcher.get_best_reward()

    def get_training_curves(self, filename=None, plot=False, use_legend=True):
        if filename is None and not plot:
            logger.warning('Please either provide filename or allow plot in get_training_curves')
        import matplotlib.pyplot as plt

        eval_metric = self.__get_training_history_metric('eval_metric', default='validation_performance')
        sign_mult = int(self.__get_training_history_metric('greater_is_better', default=True)) * 2 - 1

        plt.ylabel(eval_metric)
        plt.xlabel(self.time_attr)
        plt.title("Performance vs Training-Time in each HPO Trial")
        for task_id, task_res in self.training_history.items():
            rewards = [x[self._reward_attr] * sign_mult for x in task_res]
            x = [x[self.time_attr] for x in task_res]
            plt.plot(x, rewards, label=f'task {task_id}')
        if use_legend:
            plt.legend(loc='best')
        if filename:
            logger.info(f'Saving Training Curve in {filename}')
            plt.savefig(filename)
        if plot:
            plt.show()

    def __get_training_history_metric(self, metric, default=None):
        for _, task_res in self.training_history.items():
            if task_res and metric in task_res[0]:
                return task_res[0][metric]
        return default


    def get_best_task_id(self):
        """Get the task id that results in the best configuration/best reward.

        If there are duplicated configurations, we return the id of the first one.
        """
        best_config = self.get_best_config()
        for task_id, config in self.config_history.items():
            if best_config == config:
                return task_id
        raise RuntimeError('The best config {} is not found in config history = {}. '
                           'This should never happen!'.format(best_config, self.config_history))
