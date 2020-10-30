from typing import Optional, List
import numpy as np
import scipy.linalg as spl
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import copy

from ..autogluon.hp_ranges import HyperparameterRanges_CS
from ..datatypes.common import CandidateEvaluation
from ..datatypes.tuning_job_state import TuningJobState
from ..models.gp_model import get_internal_candidate_evaluations
from ..tuning_algorithms.default_algorithm import dictionarize_objective, \
    DEFAULT_METRIC


class ThreeHumpCamel(object):
    @property
    def search_space(self):
        return [{'min': -5.0, 'max': 5.0},
                {'min': -5.0, 'max': 5.0}]

    def evaluate(self, x1, x2):
        return 2 * x1 ** 2 - 1.05 * x1 ** 4 + x1 ** 6 / 6 + x1 * x2 + x2 ** 2


def branin_function(x1, x2, r=6):
    return (x2 - (5.1 / (4 * np.pi ** 2)) * x1 ** 2 + (5 / np.pi) * x1 - r) ** 2 + \
           10 * (1 - 1 / (8 * np.pi)) * np.cos(x1) + 10


class Branin(object):
    @property
    def search_space(self):
        return [{'min': -5.0, 'max': 10.0},
                {'min': 0.0, 'max': 15.0}]

    def evaluate(self, x1, x2):
        return branin_function(x1, x2)


class BraninWithR(Branin):
    def __init__(self, r):
        self.r = r

    def evaluate(self, x1, x2):
        return branin_function(x1, x2, r=self.r)


class Ackley(object):
    @property
    def search_space(self):
        const = 32.768
        return [{'min': -const, 'max': const},
                {'min': -const, 'max': const}]

    def evaluate(self, x1, x2):
        a = 20
        b = 0.2
        c = 2 * np.pi
        ssq = (x1 ** 2) + (x2 ** 2)
        scos = np.cos(c * x1) + np.cos(c * x2)
        return -a * np.exp(-b * np.sqrt(0.5 * ssq)) - np.exp(0.5 * scos) + \
               (a + np.exp(1))


class SimpleQuadratic(object):
    @property
    def search_space(self):
        return [{'min': 0.0, 'max': 1.0},
                {'min': 0.0, 'max': 1.0}]

    def evaluate(self, x1, x2):
        return 2 * (x1 - 0.5)**2 + (x2 - 0.5)**2


def _decode_input(x, lim):
    mn, mx = lim['min'], lim['max']
    return x * (mx - mn) + mn


def evaluate_blackbox(bb_func, inputs: np.ndarray) -> np.ndarray:
    num_dims = inputs.shape[1]
    input_list = []
    for x, lim in zip(np.split(inputs, num_dims, axis=1), bb_func.search_space):
        input_list.append(_decode_input(x, lim))
    return bb_func.evaluate(*input_list)


# NOTE: Inputs will always be in [0, 1] (so come in encoded form). They are
# only scaled to their native ranges (linearly) when evaluations of the
# blackbox are done. This avoids silly errors.
def sample_data(
        bb_cls, num_train: int, num_grid: int,
        expand_datadct: bool = True) -> dict:
    bb_func = bb_cls()
    ss_limits = bb_func.search_space
    num_dims = len(ss_limits)
    # Sample training inputs
    train_inputs = np.random.uniform(
        low=0.0, high=1.0, size=(num_train, num_dims))
    # Training targets (currently, no noise is added)
    train_targets = evaluate_blackbox(bb_func, train_inputs).reshape((-1,))
    # Inputs for prediction (regular grid)
    grids = [np.linspace(0.0, 1.0, num_grid)] * num_dims
    grids2 = tuple(np.meshgrid(*grids))
    test_inputs = np.hstack([x.reshape(-1, 1) for x in grids2])
    # Also evaluate true function on grid
    true_targets = evaluate_blackbox(bb_func, test_inputs).reshape((-1,))
    data = {
        'ss_limits': ss_limits,
        'train_inputs': train_inputs,
        'train_targets': train_targets,
        'test_inputs': test_inputs,
        'grid_shape': grids2[0].shape,
        'true_targets': true_targets}
    if expand_datadct:
        # Make sure that ours and GPy below receive exactly the same inputs
        data = expand_data(data)
    return data


def expand_data(data: dict) -> dict:
    """
    Appends derived entries to data dict, which have non-elementary types.
    """
    if 'state' not in data:
        data = copy.copy(data)
        state = data_to_state(data)
        data_internal = get_internal_candidate_evaluations(
            state, active_metric=DEFAULT_METRIC, normalize_targets=True,
            num_fantasize_samples=20)
        data['state'] = state
        data['train_inputs'] = data_internal.X
        data['train_targets_normalized'] = data_internal.y
    return data


# Recall that inputs in data are encoded, so we have to decode them to their
# native ranges for candidate_evaluations
def data_to_state(data: dict) -> TuningJobState:
    configs, cs = decode_inputs(data['train_inputs'], data['ss_limits'])
    _evaluations = [
        CandidateEvaluation(config, dictionarize_objective(y))
        for config, y in zip(configs, data['train_targets'])]
    return TuningJobState(
        hp_ranges=HyperparameterRanges_CS(cs),
        candidate_evaluations=_evaluations,
        failed_candidates=[],
        pending_evaluations=[])


def decode_inputs(inputs: np.ndarray, ss_limits) -> \
        (List[CS.Configuration], CS.ConfigurationSpace):
    cs = CS.ConfigurationSpace()
    cs_names = ['x{}'.format(i) for i in range(len(ss_limits))]
    cs.add_hyperparameters([
        CSH.UniformFloatHyperparameter(
            name=name, lower=lims['min'], upper=lims['max'])
        for name, lims in zip(cs_names, ss_limits)])
    x_mult = []
    x_add = []
    for lim in ss_limits:
        mn, mx = lim['min'], lim['max']
        x_mult.append(mx - mn)
        x_add.append(mn)
    x_mult = np.array(x_mult)
    x_add = np.array(x_add)
    configs = []
    for x in inputs:
        x_decoded = x * x_mult + x_add
        config_dct = dict(zip(cs_names, x_decoded))
        configs.append(CS.Configuration(cs, values=config_dct))
    return configs, cs


def assert_equal_candidates(candidates1, candidates2, hp_ranges, decimal=5):
    inputs1 = hp_ranges.to_ndarray_matrix(candidates1)
    inputs2 = hp_ranges.to_ndarray_matrix(candidates2)
    np.testing.assert_almost_equal(inputs1, inputs2, decimal=decimal)


def assert_equal_randomstate(randomstate1, randomstate2):
    assert str(randomstate1.get_state()) == str(randomstate2.get_state())


def compare_gpy_predict_posterior_marginals(
        test_intermediates: dict, noise_variance_gpy: Optional[float] = None):
    """
    Compares all intermediates of cholesky_computations and
    predict_posterior_marginals to using GPy and NumPy.

    Currently, this is restricted:
    - Kernel must be Matern52 with ARD
    - Mean function must be constant 0

    :param test_intermediates: Intermediates computed using our code
    :param noise_variance_gpy: Overrides noise_variance in test_intermediates.
        Use this if jitter was added during the posterior state computation.

    """
    import GPy
    # Create GPy kernel and model
    num_data = test_intermediates['features'].shape[0]
    num_dims = test_intermediates['features'].shape[1]
    lengthscales = [
        1.0 / test_intermediates['inv_bw{}'.format(i)]
        for i in range(num_dims)]
    kernel = GPy.kern.Matern52(
        num_dims,
        variance=test_intermediates['covariance_scale'],
        lengthscale=lengthscales,
        ARD=True)
    if noise_variance_gpy is None:
        noise_variance_gpy = test_intermediates['noise_variance']
    model = GPy.models.GPRegression(
        test_intermediates['features'],
        test_intermediates['targets'].reshape((-1, 1)),
        kernel=kernel, noise_var=noise_variance_gpy)
    # Compare intermediates step by step (cholesky_computations)
    kernel_mat_gpy = kernel.K(test_intermediates['features'], X2=None)
    np.testing.assert_almost_equal(
        test_intermediates['kernel_mat'], kernel_mat_gpy, decimal=5)
    sys_mat_gpy = kernel_mat_gpy + np.diag(np.ones(num_data)) * \
                  noise_variance_gpy
    np.testing.assert_almost_equal(
        test_intermediates['sys_mat'], sys_mat_gpy, decimal=5)
    chol_fact_gpy = spl.cholesky(sys_mat_gpy, lower=True)
    # Use test_intermediates['sys_mat'] instead:
    #chol_fact_gpy = spl.cholesky(test_intermediates['sys_mat'], lower=True)
    np.testing.assert_almost_equal(
        test_intermediates['chol_fact'], chol_fact_gpy, decimal=4)
    # Mean function must be constant 0
    centered_y = test_intermediates['targets'].reshape((-1, 1))
    np.testing.assert_almost_equal(
        test_intermediates['centered_y'], centered_y, decimal=9)
    pred_mat_gpy = spl.solve_triangular(chol_fact_gpy, centered_y, lower=True)
    np.testing.assert_almost_equal(
        test_intermediates['pred_mat'], pred_mat_gpy, decimal=3)
    # Compare intermediates step by step (predict_posterior_marginals)
    k_tr_te_gpy = kernel.K(test_intermediates['features'],
                           X2=test_intermediates['test_features'])
    np.testing.assert_almost_equal(
        test_intermediates['k_tr_te'], k_tr_te_gpy, decimal=5)
    linv_k_tr_te_gpy = spl.solve_triangular(chol_fact_gpy, k_tr_te_gpy, lower=True)
    np.testing.assert_almost_equal(
        test_intermediates['linv_k_tr_te'], linv_k_tr_te_gpy, decimal=4)
    pred_means_gpy = np.dot(linv_k_tr_te_gpy.T, pred_mat_gpy)
    np.testing.assert_almost_equal(
        test_intermediates['pred_means'], pred_means_gpy, decimal=4)
    k_tr_diag_gpy = kernel.Kdiag(
        test_intermediates['test_features']).reshape((-1,))
    tvec_gpy = np.sum(np.square(linv_k_tr_te_gpy), axis=0).reshape((-1,))
    pred_vars_gpy = k_tr_diag_gpy - tvec_gpy
    np.testing.assert_almost_equal(
        test_intermediates['pred_vars'], pred_vars_gpy, decimal=4)
    # Also test against GPy predict
    pred_means_gpy2, pred_vars_gpy2 = model.predict(
        test_intermediates['test_features'], include_likelihood=False)
    pred_vars_gpy2 = pred_vars_gpy2.reshape((-1,))
    np.testing.assert_almost_equal(pred_means_gpy, pred_means_gpy2, decimal=3)
    np.testing.assert_almost_equal(pred_vars_gpy, pred_vars_gpy2, decimal=3)
