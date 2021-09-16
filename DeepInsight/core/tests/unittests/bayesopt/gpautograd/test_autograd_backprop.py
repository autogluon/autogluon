import numpy
import pprint
import pytest
import autograd.numpy as anp
from autograd import grad

from autogluon.core.searcher.bayesopt.gpautograd.kernel import Matern52
from autogluon.core.searcher.bayesopt.gpautograd.mean import ScalarMeanFunction
from autogluon.core.searcher.bayesopt.gpautograd.likelihood import \
    MarginalLikelihood
from autogluon.core.searcher.bayesopt.gpautograd.gluon_blocks_helpers import \
    encode_unwrap_parameter


slack_constant = 1e-10


def _deep_copy_params(input_params):
    """
    Make a deep copy of the input arg_dict
    :param input_arg_dict:
    :return: deep copy of input_arg_dict
    """
    output_params = {}
    for name, param in input_params.items():
        output_params[name] = anp.array(param, copy=True)
    return output_params


def negative_log_posterior(
        likelihood: MarginalLikelihood, X: anp.array, Y: anp.array):
    objective_nd = likelihood(X, Y)
    # Add neg log hyperpriors, whenever some are defined
    for param_int, encoding in likelihood.param_encoding_pairs():
        if encoding.regularizer is not None:
            param = encode_unwrap_parameter(
                param_int, encoding, X)
            objective_nd = objective_nd + encoding.regularizer(
                param)
    return objective_nd

@pytest.fixture(scope='function')
def test_autograd_backprop(n, d, print_results):
    """
    Compare the gradients of the negative_log_posterior computed via 
    the method of finite difference and AutoGrad. The gradients are 
    with respect to the internal parameters.
    """
    X = anp.random.normal(size = (n, d))
    y = anp.random.normal(size = (n, 1))

    kernel = Matern52(dimension=d)
    mean = ScalarMeanFunction()
    initial_noise_variance = None
    likelihood = MarginalLikelihood(
                kernel=kernel, mean=mean,
                initial_noise_variance=initial_noise_variance)
    likelihood.initialize(force_reinit=True)
    
    params = {}
    params_ordict = likelihood.collect_params().values()
    for param in params_ordict:
        params[param.name] = param
       
    def negative_log_posterior_forward(param_dict, likelihood, X, y):
        for k in params.keys():
            params[k].set_data(param_dict[k])
        return negative_log_posterior(likelihood, X, y)
    
    params_custom = {}
    for key in params.keys():
        params_custom[key] = anp.array([anp.random.uniform()+0.3])
    params_custom_copy = _deep_copy_params(params_custom)
    
    likelihood_value = negative_log_posterior_forward(params_custom, likelihood, X, y)
    finite_diff_grad_vec = []
    for key in params.keys():
        N = negative_log_posterior_forward(params_custom, likelihood, X, y)
        params_custom_plus = params_custom.copy()
        params_custom_plus[key] *= (1 + slack_constant)
        N_plus = negative_log_posterior_forward(params_custom_plus, likelihood, X, y)
        finite_diff_grad_vec.append((N_plus-N)/(params_custom[key] * slack_constant)) 
        
    negative_log_posterior_gradient = grad(negative_log_posterior_forward)
    grad_vec = negative_log_posterior_gradient(params_custom_copy, likelihood, X, y) 
    autograd_grad_vec = list(grad_vec.values())
    if print_results:
        print('Parameter dictionary:')
        pprint.pprint(params)      
        print('\nLikelihood value: {}'.format(likelihood_value))
        print('\nGradients through finite difference:\n{}'.format(finite_diff_grad_vec))
        print('\nGradients through AutoGrad:\n{}\n'.format(autograd_grad_vec)) 
    numpy.testing.assert_almost_equal(finite_diff_grad_vec, autograd_grad_vec, decimal=3)

def test_autograd_multiple_trials():
    n, d = 20, 5
    num_of_exceptions = 0
    num_of_trials = 100
    print_results = False
    for _ in range(num_of_trials):
        try:
            test_autograd_backprop(n, d, print_results)
        except:
            num_of_exceptions += 1
    print('{} exceptions in {} trials.'.format(num_of_exceptions, num_of_trials))
