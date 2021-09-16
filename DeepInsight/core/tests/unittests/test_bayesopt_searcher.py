import time
import autogluon.core as ag
import pytest

from autogluon.core.searcher.bayesopt.utils.comparison_gpy import BraninWithR


RESOURCE_ATTR_NAME = 'epoch'
REWARD_ATTR_NAME = 'objective'


def create_train_fn(epochs=9):
    @ag.args(x1=ag.space.Real(lower=-5, upper=10),
             x2=ag.space.Real(lower=0, upper=15),
             epochs=epochs)
    def run_branin(args, reporter, **kwargs):
        for epoch in range(args.epochs):
            time.sleep(0.1)
            branin_func = BraninWithR(r=epoch)
            reporter(
                epoch=epoch + 1,
                objective=-branin_func.evaluate(args.x1, args.x2))

    return run_branin


def compute_error(df):
    return 1.0 - df["objective"]


def compute_runtime(df, start_timestamp):
        return df["time_step"] - start_timestamp


def run_bayesopt_test(sch_type):
    run_branin = create_train_fn()
    # Create scheduler and searcher:
    # First two get_config are random, the next 8 should use BO
    search_options = {
        'num_init_random': 2,
        'debug_log': True}
    if sch_type == 'fifo':
        myscheduler = ag.scheduler.FIFOScheduler(
            run_branin,
            searcher='bayesopt',
            search_options=search_options,
            num_trials=10,
            time_attr=RESOURCE_ATTR_NAME,
            reward_attr=REWARD_ATTR_NAME)
    else:
        myscheduler = ag.scheduler.HyperbandScheduler(
            run_branin,
            searcher='bayesopt',
            search_options=search_options,
            num_trials=10,
            time_attr=RESOURCE_ATTR_NAME,
            reward_attr=REWARD_ATTR_NAME,
            type=sch_type,
            grace_period=1,
            reduction_factor=3,
            brackets=1)
    # Run HPO experiment
    myscheduler.run()
    myscheduler.join_jobs()


@pytest.mark.skip(reason="This test is currently crashing the CI (#734)")
def test_bayesopt_fifo():
    run_bayesopt_test('fifo')


@pytest.mark.skip(reason="This test is currently crashing the CI (#734)")
def test_bayesopt_hyperband_stopping():
    run_bayesopt_test('stopping')


@pytest.mark.skip(reason="This test is currently crashing the CI (#734)")
def test_bayesopt_hyperband_promotion():
    run_bayesopt_test('promotion')


if __name__ == "__main__":
    test_bayesopt_fifo()
    test_bayesopt_hyperband_stopping()
    test_bayesopt_hyperband_promotion()
