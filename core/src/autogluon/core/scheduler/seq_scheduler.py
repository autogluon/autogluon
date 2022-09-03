import logging
import time
import os
from collections import OrderedDict
from copy import deepcopy
from typing import Tuple

from tqdm.auto import tqdm

from .reporter import FakeReporter
from ..searcher import searcher_factory
from ..searcher.exceptions import ExhaustedSearchSpaceError
from ..searcher.local_searcher import LocalSearcher

logger = logging.getLogger(__name__)


class LocalReporter:
    """
    Reporter implementation for LocalSequentialScheduler
    """

    def __init__(self, trial, searcher_config, training_history: dict, config_history: dict):
        self.trial = trial
        self.training_history = training_history
        self.training_history[trial] = []
        self.searcher_config = deepcopy(searcher_config)
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
                self.config_history[self.trial] = self.searcher_config
                if 'util_args' in self.searcher_config:
                    self.searcher_config.pop('util_args')

            self.last_result = result

    def terminate(self):
        pass  # compatibility


class LocalSequentialScheduler(object):
    """ Simple scheduler which schedules all HPO jobs in sequence without any parallelism.
    The next trial scheduling will be decided based on the available time left within `time_out` setting
    and average time required for a trial to complete multiplied by the fill_factor (0.95) by default to
    accommodate variance in runtimes per HPO run.

    Parameters
    ----------
    train_fn : callable
        A task launch function for training.
    resource : dict
        Computation resources. For example, `{'num_cpus':2, 'num_gpus':1}`
    searcher : str
        Searcher (get_config decisions). If str, this is passed to
        searcher_factory along with search_options.
    search_options : dict
        If searcher is str, these arguments are passed to searcher_factory.
    num_trials : int
        Maximum number of jobs run in experiment. One of `num_trials`,
        `time_out` must be given.
    time_out : float
        If given, jobs are started only until this time_out (wall clock time).
        One of `num_trials`, `time_out` must be given.
    reward_attr : str
        Name of reward (i.e., metric to maximize) attribute in data obtained
        from reporter
    time_attr : str
        Name of resource (or time) attribute in data obtained from reporter.
        This attribute is optional for FIFO scheduling, but becomes mandatory
        in multi-fidelity scheduling (e.g., Hyperband).
        Note: The type of resource must be int.
    """

    def __init__(self, train_fn, search_space, train_fn_kwargs=None, searcher='auto', reward_attr='reward', resource=None, **kwargs):
        self.train_fn = train_fn
        self.training_history = None
        self.config_history = None
        self._reward_attr = reward_attr
        self.time_attr = kwargs.get('time_attr', None)
        self.resource = resource
        self.max_reward = kwargs.get('max_reward', None)
        self.searcher: LocalSearcher = self.get_searcher_(searcher, train_fn, search_space=search_space, **kwargs)
        self.init_limits_(kwargs)
        self.train_fn_kwargs = train_fn_kwargs
        self.metadata = {
            'search_space': search_space,
            'search_strategy': self.searcher,
            'stop_criterion': {
                'time_limits': self.time_out,
                'max_reward': self.max_reward},
            'resources_per_trial': self.resource
        }

    def init_limits_(self, kwargs):
        if kwargs.get('num_trials', None) is None:
            assert kwargs.get('time_out', None) is not None, "Need stopping criterion: Either num_trials or time_out"
        self.num_trials = kwargs.get('num_trials', 9999)
        self.time_out = kwargs.get('time_out', None)
        if self.num_trials is None:
            assert self.time_out is not None, "Need stopping criterion: Either num_trials or time_out"

    def get_searcher_(self, searcher, train_fn, search_space, **kwargs) -> LocalSearcher:
        scheduler_opts = {}
        if searcher == 'auto':
            searcher = 'local_random'
            scheduler_opts = {'scheduler': 'local'}
        elif searcher == 'random':
            # FIXME: Hack to be compatible with gluoncv
            searcher = 'local_random'

        search_options = kwargs.get('search_options', None)
        if isinstance(searcher, str):
            if search_options is None:
                search_options = dict()
            _search_options = search_options.copy()
            _search_options['search_space'] = search_space
            _search_options['reward_attribute'] = self._reward_attr
            # Adjoin scheduler info to search_options, if not already done by
            # subclass
            if 'scheduler' not in _search_options:
                _search_options['scheduler'] = 'local'
            searcher = searcher_factory(searcher, **{**scheduler_opts, **_search_options})
        else:
            assert isinstance(searcher, LocalSearcher)
        return searcher

    def run(self, **kwargs):
        """Run multiple trials given specific time and trial numbers limits.
        """
        self.searcher.configure_scheduler(self)

        self.training_history = OrderedDict()
        self.config_history = OrderedDict()

        failure_count = 0
        trial_count = 0
        trials_total_time = 0
        min_failure_threshold = 5
        failure_rate_threshold = 0.8
        time_start = time.time()

        r = range(self.num_trials)
        for i in (tqdm(r) if self.num_trials < 1000 else r):
            trial_start_time = time.time()
            try:
                is_failed, result = self.run_trial(task_id=i)
            except ExhaustedSearchSpaceError:
                break
            except Exception:
                # TODO: Add special exception type when there are no more new configurations to try (exhausted search space)
                logger.log(30, f'\tWARNING: Encountered unexpected exception during trial {i}, stopping HPO early.')
                logger.exception('Detailed Traceback:')  # TODO: Avoid logging if verbosity=0
                break
            trial_end_time = time.time()

            trial_count += 1
            if is_failed:
                failure_count += 1
            else:
                trials_total_time += trial_end_time - trial_start_time

            if self.max_reward and self.get_best_reward() >= self.max_reward:
                logger.log(20, f'\tStopping HPO: Max reward reached')
                break

            if failure_count >= min_failure_threshold and (failure_count / trial_count) >= failure_rate_threshold:
                logger.warning(f'Warning: Detected a large trial failure rate: '
                               f'{failure_count}/{trial_count} attempted trials failed ({round((failure_count / trial_count) * 100, 1)}%)! '
                               f'Stopping HPO early due to reaching failure threshold ({round(failure_rate_threshold*100, 1)}%).\n'
                               f'\tFailures may be caused by invalid configurations within the provided search space.')
                break

            if self.time_out is not None:
                avg_trial_run_time = 0 if trial_count == failure_count else trials_total_time / (trial_count - failure_count)
                if not self.has_enough_time_for_trial_(self.time_out, time_start, trial_start_time, trial_end_time, avg_trial_run_time):
                    logger.log(20, f'\tStopping HPO to satisfy time limit...')
                    break

    @classmethod
    def has_enough_time_for_trial_(cls, time_out, time_start, trial_start_time, trial_end_time, avg_trial_run_time, fill_factor=0.95):
        """
        Checks if the remaining time is enough to run another trial.

        Parameters
        ----------
        time_out total
            timeout in m
        time_start
            trials start time
        trial_start_time
            last trial start time
        trial_end_time
            last trial end time
        avg_trial_run_time
            running average of all trial runs
        fill_factor: float
            discount of `avg_trial_run_time` allowed for a next trial. Default is 0.95 of `avg_trial_run_time`

        Returns
        -------
            True if there is enough time to run another trial give runs statistics and remaining time

        """
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

    def run_trial(self, task_id=0) -> Tuple[bool, dict]:
        """
        Start a trial with a given task_id

        Parameters
        ----------
        task_id
            task

        Returns
        -------
        is_failed: bool
            True if task completed successfully
        trial_start_time
            Trial start time
        trial_end_time
            Trial end time

        """
        new_searcher_config = self.searcher.get_config()
        searcher_config = deepcopy(self.metadata['search_space'])
        searcher_config.update(new_searcher_config)
        reporter = LocalReporter(task_id, searcher_config, self.training_history, self.config_history)
        return self.run_job_(task_id, searcher_config, reporter)

    def run_job_(self, task_id, searcher_config, reporter):
        args = dict()
        if self.train_fn_kwargs is not None:
            # TODO: Consider avoiding deepcopy and just do shallow copy,
            #  Risk is that it will allow values in self.train_fn_kwargs to be altered by HPO trials, causing early trials to alter later trials.
            train_fn_kwargs = deepcopy(self.train_fn_kwargs)
        else:
            train_fn_kwargs = dict()
        args.update(searcher_config)

        args['task_id'] = task_id
        self.searcher.register_pending(searcher_config)
        is_failed = False
        try:
            result = self.train_fn(args, reporter=reporter, **train_fn_kwargs)
            if type(reporter) is not FakeReporter:
                if reporter.last_result:
                    self.searcher.update(config=searcher_config, **reporter.last_result)
                else:
                    is_failed = True
        except Exception as e:
            logger.error(f'Exception during a trial: {e}')
            self.searcher.evaluation_failed(config=searcher_config)
            reporter(traceback=e)
            is_failed = True
            result = {'traceback': str(e)}
        return is_failed, result

    def run_with_config(self, config):
        """Run with config for final fit.
        It launches a single training trial under any fixed values of the hyperparameters.
        For example, after HPO has identified the best hyperparameter values based on a hold-out dataset,
        one can use this function to retrain a model with the same hyperparameters on all the available labeled data
        (including the hold out set). It can also returns other objects or states.
        """
        is_failed, result = self.run_job_('run_with_config', config, FakeReporter())
        return result

    def join_jobs(self, timeout=None):
        pass  # Compatibility

    def get_best_config(self):
        """Get the best configuration from the finished jobs.
        """
        # TODO: Consider passing the metadata search space to searcher to avoid having to do this
        searcher_config = deepcopy(self.metadata['search_space'])
        searcher_config.update(self.searcher.get_best_config())
        return searcher_config

    def get_best_reward(self):
        """Get the best reward from the finished jobs.
        """
        return self.searcher.get_best_reward()

    def get_training_curves(self, filename=None, plot=False, use_legend=True):
        """Get Training Curves
        """
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
            file_dir = os.path.split(os.path.abspath(filename))[0]
            if not os.path.exists(file_dir):
                os.makedirs(file_dir)
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
