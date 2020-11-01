from autogluon.core.decorator import sample_config
from autogluon.core.scheduler.resource import get_cpu_count, get_gpu_count
from autogluon.core.task.base import BaseTask
import autogluon.core as ag

@ag.args()
def _train_fn_bad(args, reporter):
    raise NotImplementedError

@ag.args()
def _train_fn_good(args, reporter):
    reporter(epoch=1, accuracy=0)

class TestTask(BaseTask):
    def __init__(self, fn):
        self.fn = fn
        self._config = {}
        nthreads_per_trial = get_cpu_count()
        ngpus_per_trial = get_gpu_count()
        self._config['search_strategy'] = self._config.get('search_strategy', 'random')
        self._config['scheduler_options'] = {
            'resource': {'num_cpus': nthreads_per_trial, 'num_gpus': ngpus_per_trial},
            'num_trials': self._config.get('num_trials', 2),
            'time_out': self._config.get('time_limits', 60 * 60),
            'time_attr': 'epoch',
            'reward_attr': 'accuracy',
            'searcher': self._config.get('search_strategy', {}),
            'search_options': self._config.get('search_options', None)}
        if self._config['search_strategy'] == 'hyperband':
            self._config['scheduler_options'].update({
                'searcher': 'random',
                'max_t': self._config.get('epochs', 50),
                'grace_period': self._config.get('grace_period',  100)})

    def fit(self):
        config = {'lr': ag.Categorical(0.1, 0.2, 0.3)}
        self.fn.register_args(**config)
        results = self.run_fit(self.fn, self._config['search_strategy'],
                               self._config['scheduler_options'])

def test_valid_fn():
    task = TestTask(_train_fn_good)
    task.fit()

def test_invalid_fn():
    # should catch, print but won't raise
    task = TestTask(_train_fn_bad)
    task.fit()
