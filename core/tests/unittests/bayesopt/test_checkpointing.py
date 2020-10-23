import numpy as np
import pickle

from autogluon.core.searcher.bayesopt.autogluon.searcher_factory import \
    gp_fifo_searcher_factory, gp_fifo_searcher_defaults
from autogluon.core.searcher.bayesopt.tuning_algorithms.default_algorithm \
    import DEFAULT_METRIC
from autogluon.core.searcher.bayesopt.utils.comparison_gpy import \
    Ackley, sample_data, assert_equal_candidates, assert_equal_randomstate


def test_pickle_gp_fifo_searcher():
    random_seed = 894623209
    # This data is used below
    _, searcher_options, _ = gp_fifo_searcher_defaults()
    num_data = searcher_options['num_init_random'] + 2
    num_pending = 2
    data = sample_data(Ackley, num_train=num_data + num_pending, num_grid=5)
    # Create searcher1 using default arguments
    searcher_options['configspace'] = data['state'].hp_ranges.config_space
    searcher_options['scheduler'] = 'fifo'
    searcher_options['random_seed'] = random_seed
    searcher1 = gp_fifo_searcher_factory(**searcher_options)
    # Feed searcher1 with some data
    for eval in data['state'].candidate_evaluations[:num_data]:
        reward = searcher1.map_reward.reverse(eval.metrics[DEFAULT_METRIC])
        searcher1.update(eval.candidate, reward)
    # Calling next_config is forcing a GP hyperparameter update
    next_config = searcher1.get_config()
    # Register some pending evaluations
    for eval in data['state'].candidate_evaluations[-num_pending:]:
        searcher1.register_pending(eval.candidate)
    # Pickle mutable state of searcher1
    pkl_state = pickle.dumps(searcher1.get_state())
    # Clone searcher2 from mutable state
    searcher2 = gp_fifo_searcher_factory(**searcher_options)
    searcher2 = searcher2.clone_from_state(pickle.loads(pkl_state))
    # At this point, searcher1 and searcher2 should be essentially the same
    # Compare model parameters
    params1 = searcher1.get_params()
    params2 = searcher2.get_params()
    for k, v1 in params1.items():
        v2 = params2[k]
        np.testing.assert_almost_equal(
            np.array([v1]), np.array([v2]), decimal=4)
    # Compare states
    state1 = searcher1.state_transformer.state
    state2 = searcher2.state_transformer.state
    hp_ranges = state1.hp_ranges
    assert_equal_candidates(
        [x.candidate for x in state1.candidate_evaluations],
        [x.candidate for x in state2.candidate_evaluations], hp_ranges,
        decimal=5)
    eval_targets1 = np.array([
        x.metrics[DEFAULT_METRIC] for x in state1.candidate_evaluations])
    eval_targets2 = np.array([
        x.metrics[DEFAULT_METRIC] for x in state1.candidate_evaluations])
    np.testing.assert_almost_equal(eval_targets1, eval_targets2, decimal=5)
    assert_equal_candidates(
        state1.pending_candidates, state2.pending_candidates, hp_ranges,
        decimal=5)
    # Compare random_state, random_generator state
    assert_equal_randomstate(searcher1.random_state, searcher2.random_state)


if __name__ == "__main__":
    test_pickle_gp_fifo_searcher()
