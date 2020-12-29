from functools import partial

from ray import tune

from autogluon.core.space import *
from autogluon.core.utils.edict import EasyDict


class RayTuneScheduler(object):

    def __init__(self, task_fn, **tune_args):
        self.task_fn = task_fn
        self.tune_args = tune_args

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

    def run_job(self, **kwargs):
        result = tune.run(
            partial(self.train_fn_wrapper, self.task_fn.f),
            config=self.wrap_space(self.task_fn.kwvars),
            **self.tune_args
        )
        return result
