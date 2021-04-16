import autogluon.core as ag
import numpy as np
import pytest

ACTIVE_METRIC_NAME = 'brainin_objective'
CONSTRAINT_METRIC_NAME = 'brainin_constraint'
REWARD_ATTR_NAME = 'objective'
CONSTRAINT_ATTR_NAME = 'constraint_metric'


def brainin_with_constraint(x1, x2, constraint_offset):
    evaluation_dict = {}
    r = 6
    objective_value = (x2 - (5.1 / (4 * np.pi ** 2)) * x1 ** 2 +
                       (5 / np.pi) * x1 - r) ** 2 + 10 * (1 - 1 / (8 * np.pi)) * np.cos(x1) + 10
    evaluation_dict[ACTIVE_METRIC_NAME] = objective_value
    # Feasible iff x1 <= constraint_offset / 2.0
    evaluation_dict[CONSTRAINT_METRIC_NAME] = x1 * 2.0 - constraint_offset
    return evaluation_dict


def create_train_fn_constraint(constraint_offset):
    @ag.args(x1=ag.space.Real(lower=-5, upper=10),
             x2=ag.space.Real(lower=0, upper=15))
    def run_branin(args, reporter):
        branin_eval = brainin_with_constraint(args.x1, args.x2, constraint_offset)
        reporter(objective=-branin_eval[ACTIVE_METRIC_NAME],
                 constraint_metric=branin_eval[CONSTRAINT_METRIC_NAME])
    return run_branin


def run_bayesopt_multi_output_test(searcher, constraint_offset):
    run_branin = create_train_fn_constraint(constraint_offset)
    # Create scheduler and searcher:
    # First two get_config are random, the next 8 use constrained BO
    random_seed = 123
    search_options = {
        'random_seed': random_seed,
        'num_fantasy_samples': 5,
        'num_init_random': 2,
        'debug_log': True}
    myscheduler = ag.scheduler.FIFOScheduler(
        run_branin,
        searcher=searcher,
        search_options=search_options,
        num_trials=10,
        reward_attr=REWARD_ATTR_NAME,
        constraint_attr=CONSTRAINT_ATTR_NAME,
    )
    # Run HPO experiment
    myscheduler.run()
    myscheduler.join_jobs()


@pytest.mark.skip(reason="This test is currently crashing the CI (#734)")
def test_constrained_bayesopt_loose_constraint_fifo():
    run_bayesopt_multi_output_test('constrained_bayesopt', 20.0)


@pytest.mark.skip(reason="This test is currently crashing the CI (#734)")
def test_constrained_bayesopt_strict_constraint_fifo():
    run_bayesopt_multi_output_test('constrained_bayesopt', 1.0)


@pytest.mark.skip(reason="This test is currently crashing the CI (#734)")
def test_standard_bayesopt_strict_constraint_fifo():
    run_bayesopt_multi_output_test('bayesopt', 1.0)


if __name__ == "__main__":
    test_constrained_bayesopt_loose_constraint_fifo()
    test_constrained_bayesopt_strict_constraint_fifo()
    test_standard_bayesopt_strict_constraint_fifo()
