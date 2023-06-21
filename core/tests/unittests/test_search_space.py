from autogluon.common import space
from autogluon.core.scheduler import LocalSequentialScheduler


def test_search_space():
    search_space = dict(
        a=space.Real(1e-3, 1e-2, log=True),
        b=space.Real(1e-3, 1e-2),
        c=space.Int(1, 10),
        d=space.Categorical("a", "b", "c", "d"),
    )

    def train_fn(args, reporter):
        a, b, c, d = args["a"], args["b"], args["c"], args["d"]
        assert 1e-2 >= a >= 1e-3
        assert 1e-2 >= b >= 1e-3
        assert 10 >= c >= 1
        assert d in ["a", "b", "c", "d"]
        reporter(epoch=1, reward=0)

    scheduler = LocalSequentialScheduler(train_fn, search_space=search_space, num_trials=10, time_attr="epoch")
    scheduler.run()
    scheduler.join_jobs()


def test_search_space_dot_key():
    search_space = {"model.name": space.Categorical("mxnet", "pytorch")}

    def train_fn(args, reporter):
        assert args["model.name"] == "mxnet" or args["model.name"] == "pytorch"

    scheduler = LocalSequentialScheduler(train_fn, search_space=search_space, num_trials=2)
    scheduler.run()
    scheduler.join_jobs()
