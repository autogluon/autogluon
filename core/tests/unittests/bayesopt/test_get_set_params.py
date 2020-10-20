import numpy as np

from autogluon.core.searcher.bayesopt.autogluon.searcher_factory import \
    gp_multifidelity_searcher_factory, gp_multifidelity_searcher_defaults
from autogluon.core.searcher.bayesopt.utils.comparison_gpy import \
    Ackley, sample_data


def test_params_gp_multifidelity():
    # Create GP multifidelity searcher, including a GP surrogate model
    _, searcher_options, _ = gp_multifidelity_searcher_defaults()
    searcher_options['gp_resource_kernel'] = 'exp-decay-combined'
    # Note: We are lazy here, we just need the config_space
    data = sample_data(Ackley, num_train=5, num_grid=5)
    searcher_options['configspace'] = data['state'].hp_ranges.config_space
    searcher_options['scheduler'] = 'hyperband_stopping'
    searcher_options['min_reward'] = 0.
    searcher_options['min_epochs'] = 1
    searcher_options['max_epochs'] = 27
    searcher_options['reward_attribute'] = 'accuracy'
    searcher_options['resource_attribute'] = 'epoch'
    searcher = gp_multifidelity_searcher_factory(**searcher_options)
    # Set parameters
    params = {
        'noise_variance': 0.01,
        'kernel_alpha': 9.0,
        'kernel_mean_lam': 0.25,
        'kernel_gamma': 0.75,
        'kernel_delta': 0.125,
        'kernel_kernelx_inv_bw0': 0.11,
        'kernel_kernelx_inv_bw1': 11.0,
        'kernel_kernelx_covariance_scale': 5.5,
        'kernel_meanx_mean_value': 1e-5}
    searcher.set_params(params)
    # Get parameters: Must be the same
    params2 = searcher.get_params()
    assert len(params) == len(params2), (params, params2)
    for k, v in params.items():
        assert k in params2, (k, params, params2)
        v2 = params2[k]
        np.testing.assert_almost_equal(
            [v], [v2], decimal=6, err_msg='key={}'.format(k))


if __name__ == "__main__":
    test_params_gp_multifidelity()
