import autogluon.core as ag
import logging
import numpy as np
import time

from autogluon.core.searcher.bayesopt.autogluon.hp_ranges import \
    HyperparameterRanges_CS
from autogluon.core.searcher.bayesopt.utils.comparison_gpy import Branin
from autogluon.core.searcher.gp_searcher import _to_config_cs
from autogluon.core import Task

logger = logging.getLogger(__name__)


@ag.args(
    x1=ag.space.Real(-5.0, 10.0),
    x2=ag.space.Real(0.0, 15.0))
def branin_fn(args, reporter, **kwargs):
    func = Branin()
    accuracy = -func.evaluate(args.x1, args.x2)
    reporter(accuracy=accuracy)


# The dependence on epoch is totally made up here, not suitable to benchmark
# Hyperband.
# We also sample a sleep time per epoch in 0.1 * [0.5, 1.5]. The random
# seed for this is generated from the input in a way that if two inputs are
# very close, they map to the same seed.
@ag.args(
    x1=ag.space.Real(-5.0, 10.0),
    x2=ag.space.Real(0.0, 15.0),
    epochs=9)
def branin_epochs_fn(args, reporter):
    func = Branin()
    if 'scheduler' in args and 'resume_from' in args.scheduler:
        resume_from = args.scheduler.resume_from
    else:
        resume_from = 1
    # Sample sleep time per epoch
    x1 = (args.x1 + 5) / 15
    x2 = args.x2 / 15
    local_seed = int(np.floor(x1 * x2 * 10000))
    random_state = np.random.RandomState(local_seed)
    time_per_epoch = random_state.uniform(0.05, 0.15)
    #print("[{}, {}]: seed={}, time={}".format(
    #    args.x1, args.x2, local_seed, time_per_epoch))
    for epoch in range(resume_from, args.epochs + 1):
        factor = epoch / args.epochs
        x1 = (args.x1 - 2.5) * factor + 2.5
        x2 = (args.x2 - 7.5) * factor + 7.5
        accuracy = -func.evaluate(x1, x2)
        time.sleep(time_per_epoch)
        reporter(accuracy=accuracy, epoch=epoch)


def to_input(config_dict: dict,
             hp_ranges: HyperparameterRanges_CS) -> np.ndarray:
    config = _to_config_cs(hp_ranges.config_space, config_dict)
    return hp_ranges.to_ndarray(config)


def _to_tuple(config):
    keys = sorted(config.keys())
    return tuple(config[k] for k in keys)


# Pretty silly, slow code to do this:
def remove_duplicate_configs(config_history, task_id):
    unique_configs = []
    configs_so_far = set()
    for i in range(len(config_history)):
        key = str(i + task_id)
        config = config_history[key]
        _config = _to_tuple(config)
        if _config not in configs_so_far:
            unique_configs.append(config)
            configs_so_far.add(_config)
    return unique_configs


def assert_same_config_history(
        results1, results2, num_trials, hp_ranges, decimal=5):
    config_history1 = results1['config_history']
    config_history2 = results2['config_history']
    assert len(config_history1) == num_trials
    assert len(config_history2) == num_trials
    config_history1 = remove_duplicate_configs(
        config_history1, results1['task_id'])
    config_history2 = remove_duplicate_configs(
        config_history2, results2['task_id'])
    for i in range(min(len(config_history1), len(config_history2))):
        input1 = to_input(config_history1[i], hp_ranges)
        input2 = to_input(config_history2[i], hp_ranges)
        np.testing.assert_almost_equal(
            input1, input2, decimal=decimal,
            err_msg='Configs different for i = {}'.format(i))


def print_config_history(config_history, task_id):
    for i in range(len(config_history)):
        key = str(i + task_id)
        print("{}: {}".format(i, config_history[key]))


def test_resume_fifo_random():
    random_seed = 623478423
    num_trials1 = 10
    num_trials2 = 20
    exper_type = 'fifo_random'
    checkpoint_fname = 'tests/unittests/checkpoint_{}.ag'.format(exper_type)
    search_options = {'random_seed': random_seed}

    # First experiment: Two phases, with resume
    task_id = Task.TASK_ID.value
    scheduler1 = ag.scheduler.FIFOScheduler(
        branin_fn,
        searcher='random',
        search_options=search_options,
        checkpoint=checkpoint_fname,
        num_trials=num_trials1,
        reward_attr='accuracy')
    logger.info("Running [{} - two phases]: num_trials={} and checkpointing".format(
        exper_type, num_trials1))
    scheduler1.run()
    scheduler1.join_jobs()
    scheduler2 = ag.scheduler.FIFOScheduler(
        branin_fn,
        searcher='random',
        search_options=search_options,
        checkpoint=checkpoint_fname,
        num_trials=num_trials2,
        reward_attr='accuracy',
        resume=True)
    logger.info("Running [{} - two phases]: Resume from checkpoint, num_trials={}".format(
        exper_type, num_trials2))
    scheduler2.run()
    scheduler2.join_jobs()
    searcher = scheduler2.searcher
    results1 = {
        'task_id': task_id,
        'config_history': scheduler2.config_history,
        'best_reward': searcher.get_best_reward(),
        'best_config': searcher.get_best_config()}

    # Second experiment: Just one phase
    task_id = Task.TASK_ID.value
    scheduler3 = ag.scheduler.FIFOScheduler(
        branin_fn,
        searcher='random',
        search_options=search_options,
        checkpoint=None,
        num_trials=num_trials2,
        reward_attr='accuracy')
    logger.info("Running [{} - one phase]: num_trials={}".format(
        exper_type, num_trials2))
    scheduler3.run()
    scheduler3.join_jobs()
    searcher = scheduler3.searcher
    results2 = {
        'task_id': task_id,
        'config_history': scheduler3.config_history,
        'best_reward': searcher.get_best_reward(),
        'best_config': searcher.get_best_config()}

    hp_ranges = HyperparameterRanges_CS(branin_fn.cs)
    assert_same_config_history(
        results1, results2, num_trials2, hp_ranges)


def test_resume_hyperband_random():
    random_seed = 623478423
    num_trials1 = 20
    num_trials2 = 40
    search_options = {'random_seed': random_seed}
    scheduler_options = {
        'reward_attr': 'accuracy',
        'time_attr': 'epoch',
        'max_t': 9,
        'grace_period': 1,
        'reduction_factor': 3,
        'brackets': 1}

    # Note: The difficulty with HB promotion is that when configs are
    # promoted, they appear more than once in config_history (which is
    # indexed by task_id). This can lead to differences between the two
    # and one phase runs. What matters is that the same configs are
    # proposed.
    for hp_type in ['stopping', 'promotion']:
        exper_type = 'hyperband_{}_random'.format(hp_type)
        checkpoint_fname = 'tests/unittests/checkpoint_{}.ag'.format(exper_type)
        # First experiment: Two phases, with resume
        task_id = Task.TASK_ID.value
        scheduler1 = ag.scheduler.HyperbandScheduler(
            branin_epochs_fn,
            searcher='random',
            search_options=search_options,
            checkpoint=checkpoint_fname,
            num_trials=num_trials1,
            type=hp_type,
            **scheduler_options)
        logger.info("Running [{} - two phases]: num_trials={} and checkpointing".format(
            exper_type, num_trials1))
        scheduler1.run()
        scheduler1.join_jobs()
        scheduler2 = ag.scheduler.HyperbandScheduler(
            branin_epochs_fn,
            searcher='random',
            search_options=search_options,
            checkpoint=checkpoint_fname,
            num_trials=num_trials2,
            type=hp_type,
            resume=True,
            **scheduler_options)
        logger.info("Running [{} - two phases]: Resume from checkpoint, num_trials={}".format(
            exper_type, num_trials2))
        scheduler2.run()
        scheduler2.join_jobs()
        searcher = scheduler2.searcher
        results1 = {
            'task_id': task_id,
            'config_history': scheduler2.config_history,
            'best_reward': searcher.get_best_reward(),
            'best_config': searcher.get_best_config()}
        # DEBUG
        #print_config_history(results1['config_history'], task_id)

        # Second experiment: Just one phase
        task_id = Task.TASK_ID.value
        scheduler3 = ag.scheduler.HyperbandScheduler(
            branin_epochs_fn,
            searcher='random',
            search_options=search_options,
            checkpoint=None,
            num_trials=num_trials2,
            type=hp_type,
            **scheduler_options)
        logger.info("Running [{} - one phase]: num_trials={}".format(
            exper_type, num_trials2))
        scheduler3.run()
        scheduler3.join_jobs()
        searcher = scheduler3.searcher
        results2 = {
            'task_id': task_id,
            'config_history': scheduler3.config_history,
            'best_reward': searcher.get_best_reward(),
            'best_config': searcher.get_best_config()}
        # DEBUG
        #print_config_history(results2['config_history'], task_id)

        hp_ranges = HyperparameterRanges_CS(branin_epochs_fn.cs)
        assert_same_config_history(
            results1, results2, num_trials2, hp_ranges)


if __name__ == "__main__":
    test_resume_fifo_random()
    test_resume_hyperband_random()
