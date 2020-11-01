import os
from typing import List, Optional
import numpy as np
import matplotlib.pyplot as plt
import pickle
import tempfile

from autogluon.core.searcher.bayesopt.tuning_algorithms.default_algorithm import \
    DEFAULT_METRIC
from autogluon.core.searcher.bayesopt.models.gp_model import \
    GaussProcSurrogateModel
from autogluon.core.searcher.bayesopt.gpautograd.gp_regression import \
    GaussianProcessRegression
from autogluon.core.searcher.bayesopt.gpautograd.kernel import Matern52
from autogluon.core.searcher.bayesopt.gpautograd.mean import ZeroMeanFunction
from autogluon.core.searcher.bayesopt.gpautograd.constants import \
     INITIAL_COVARIANCE_SCALE, INITIAL_INVERSE_BANDWIDTHS, \
     INVERSE_BANDWIDTHS_LOWER_BOUND, INVERSE_BANDWIDTHS_UPPER_BOUND, \
     COVARIANCE_SCALE_LOWER_BOUND, COVARIANCE_SCALE_UPPER_BOUND, \
     INITIAL_NOISE_VARIANCE, NOISE_VARIANCE_LOWER_BOUND, \
     NOISE_VARIANCE_UPPER_BOUND, OptimizationConfig, DEFAULT_OPTIMIZATION_CONFIG
#from autogluon.core.searcher.bayesopt.utils.comparison_gpy import Branin, \
#    ThreeHumpCamel, Ackley, sample_data, compare_gpy_predict_posterior_marginals
from autogluon.core.searcher.bayesopt.utils.comparison_gpy import expand_data
from autogluon.core.utils.files import download


# Note: At present, the surrogate model is fixed
# Note: Instead of a ScalarMeanFunction, we use a ZeroMeanFunction here. This
# is because it is too annoying to make GPy use such a mean function
def fit_predict_ours(
        data: dict, random_seed: int,
        optimization_config: OptimizationConfig,
        test_intermediates: Optional[dict] = None) -> dict:
    # Create surrogate model
    num_dims = len(data['ss_limits'])
    _gpmodel = GaussianProcessRegression(
        kernel=Matern52(num_dims, ARD=True),
        mean=ZeroMeanFunction(),  # Instead of ScalarMeanFunction
        optimization_config=optimization_config,
        random_seed=random_seed,
        test_intermediates=test_intermediates)
    model = GaussProcSurrogateModel(
        data['state'], DEFAULT_METRIC, random_seed, _gpmodel,
        fit_parameters=True, num_fantasy_samples=20)
    model_params = model.get_params()
    print('Hyperparameters: {}'.format(model_params))
    # Prediction
    predictions = model.predict(data['test_inputs'])[0]
    return {
        'means': predictions['mean'],
        'stddevs': predictions['std']}


# Note: this function is not ran by the tests so that we do not have to require
# GPy as a dependency of the tests. With GPy 1.9.9, it can be executed to generate
# the numerical values that are tested in the main testing function `test_comparison_gpy`
def fit_predict_gpy(
        data: dict, random_seed: int,
        optimization_config: OptimizationConfig) -> dict:
    import GPy  # Needs GPy to be installed
    assert GPy.__version__ == "1.9.9"
    # Create surrogate model
    # Note: Matern52 in GPy uses lengthscales, while Matern52 here uses inverse
    # lengthscales (or "bandwidths"), so have to apply 1 / x.
    # We use a zero mean function as it isn't straightforward to make GPy use 
    # something else.
    num_dims = len(data['ss_limits'])
    _kernel = GPy.kern.Matern52(
        num_dims,
        variance=INITIAL_COVARIANCE_SCALE,
        lengthscale=1.0 / INITIAL_INVERSE_BANDWIDTHS,
        ARD=True)
    _kernel.lengthscale.constrain_bounded(
        1.0 / INVERSE_BANDWIDTHS_UPPER_BOUND,
        1.0 / INVERSE_BANDWIDTHS_LOWER_BOUND, warning=False)
    _kernel.variance.constrain_bounded(
        COVARIANCE_SCALE_LOWER_BOUND, COVARIANCE_SCALE_UPPER_BOUND,
        warning=False)
    # Normalize targets to mean 0, variance 1 (this is done in our code
    # internally)
    targets_mean = np.mean(data['train_targets'])
    targets_std = max(np.std(data['train_targets']), 1e-8)
    targets_normalized = data['train_targets_normalized']
    model = GPy.models.GPRegression(
        data['train_inputs'], targets_normalized.reshape((-1, 1)),
        kernel=_kernel, noise_var=INITIAL_NOISE_VARIANCE)
    model.likelihood.variance.constrain_bounded(
        NOISE_VARIANCE_LOWER_BOUND, NOISE_VARIANCE_UPPER_BOUND,
        warning=False)
    # Note: We could also set hyperpriors somehow, using model.priors. But
    # this is pretty undocumented!
    # Fit hyperparameters
    # Note: Should set optimization_config.lbfgs_tol here, but don't know how
    np.random.seed(random_seed)  # Seed plays very different role here. Whatever...
    verbose = optimization_config.verbose
    model.optimize_restarts(
        num_restarts=optimization_config.n_starts,
        optimizer='bfgs',
        max_iters=optimization_config.lbfgs_maxiter,
        verbose=verbose)
    # Print hyperparameter values
    print(model)
    print('\n' + str(_kernel.lengthscale))
    # Prediction (have to be rescaled to undo normalization)
    means, vars = model.predict(
        data['test_inputs'], include_likelihood=False)
    means = means * targets_std + targets_mean
    stddevs = np.sqrt(vars) * targets_std
    return {
        'means': means,
        'stddevs': stddevs}


def _plot_comparison(y_list: List[np.ndarray]):
    yshape = y_list[0].shape
    assert all(y.shape == yshape for y in y_list)
    assert len(yshape) == 2, "Can only do 2D plots"
    min_val = min([np.min(y) for y in y_list])
    separator = np.ones((yshape[0], 5)) * min_val
    ys_and_seps = [x for l in zip(y_list, [separator] * len(y_list)) for x in l]
    ys_and_seps = ys_and_seps[:-1]
    plt.imshow(np.concatenate(ys_and_seps, axis=1))
    plt.colorbar()


# Note: this function is not ran by the tests but can be executed separately
# to get a visual interpretation, see `test_comparison_gpy`
def plot_predictions(
        data: dict, pred_ours: dict, pred_gpy: dict, title: str):
    grid_shape = data['grid_shape']
    plt.title(title)
    for i, key in enumerate(('means', 'stddevs')):
        plt.subplot(2, 1, i + 1)
        lst = [pred_ours[key].reshape(grid_shape),
               pred_gpy[key].reshape(grid_shape)]
        if i == 0:
            lst.insert(1, data['true_targets'].reshape(grid_shape))
        _plot_comparison(lst)
        plt.title(key)
    plt.show()


SRC_URL = 'https://autogluon.s3.amazonaws.com'


def download_pickle_file(fname):
    trg_path = tempfile.mkdtemp()
    trg_fname = os.path.join(trg_path, 'numcomp', fname)
    if not os.path.exists(trg_fname):
        download(os.path.join(SRC_URL, 'numcomp', fname), path=trg_fname)
    with open(trg_fname, 'rb') as handle:
        data = pickle.load(handle)
    if 'train_inputs' in data:
        data = expand_data(data)
    return data


# Main testing function
def test_comparison_gpy():
    optimization_config = DEFAULT_OPTIMIZATION_CONFIG
    fname_msk = '{}_{}_{}.pickle'
    num_train = 200
    num_grid = 200

    do_branin = True
    do_threehump = False
    do_ackley = True
    # Uncomment the lines between --- to re-generate data and GPy predictions
    # (assumes you have GPy available and verifying the version assertion in
    # `fit_predict_gpy`).
    trg_path = os.path.join(tempfile.mkdtemp(), 'numcomp')

    if do_branin:
        random_seed = 894623209
        data_name = 'branin'
        fname = fname_msk.format(data_name, num_train, 'data')
        branin = download_pickle_file(fname)
        fname = fname_msk.format(data_name, num_train, 'gpy')
        branin_gpy = download_pickle_file(fname)
        # --------------------------------------------------------------------
        #bb_cls = Branin
        #branin = sample_data(bb_cls, num_train, num_grid,
        #    expand_datadct=False)
        #print("Storing files to " + trg_path)
        #os.makedirs(trg_path, exist_ok=True)
        #fname = os.path.join(
        #    trg_path, fname_msk.format(data_name, num_train, 'data'))
        #with open(fname, 'wb') as handle:
        #    pickle.dump(branin, handle, protocol=pickle.HIGHEST_PROTOCOL)
        #branin = expand_data(branin)
        #
        #branin_gpy = fit_predict_gpy(branin, random_seed, optimization_config)
        #fname = os.path.join(
        #    trg_path, fname_msk.format(data_name, num_train, 'gpy'))
        #with open(fname, 'wb') as handle:
        #    pickle.dump(branin_gpy, handle, protocol=pickle.HIGHEST_PROTOCOL)
        # --------------------------------------------------------------------

        # test_intermediates = dict()  # DEBUG
        test_intermediates = None
        branin_ours = fit_predict_ours(
            branin, random_seed, optimization_config,
            test_intermediates=test_intermediates)
        # DEBUG:
        # compare_gpy_predict_posterior_marginals(
        #     test_intermediates, noise_variance_gpy=None)
        # END DEBUG

        # If you want a visual result, uncomment the following lines
        # title = "Branin, num_train={}".format(num_train)
        # plot_predictions(branin, branin_ours, branin_gpy, title)

        # Branin - means
        sse = sum((branin_gpy['means'].reshape(-1) - branin_ours['means'])**2)
        num = branin_ours['means'].shape[0]
        branin_means_rmse = np.sqrt(sse / num)
        # print('branin_means = {}'.format(branin_means_rmse))
        assert branin_means_rmse <= 5e-3
        # Branin - stds
        sse = sum((branin_gpy['stddevs'].reshape(-1) - branin_ours['stddevs'])**2)
        num = branin_ours['stddevs'].shape[0]
        branin_stds_rmse = np.sqrt(sse / num)
        # print('branin_stds = {}'.format(branin_stds_rmse))
        assert branin_stds_rmse <= 2e-2

    # NOTE: Fails with differences in the predictive stddevs!
    if do_threehump:
        random_seed = 54654209
        data_name = 'threehump'
        fname = fname_msk.format(data_name, num_train, 'data')
        threehump = download_pickle_file(fname)
        fname = fname_msk.format(data_name, num_train, 'gpy')
        threehump_gpy = download_pickle_file(fname)
        # --------------------------------------------------------------------
        #bb_cls = ThreeHumpCamel
        #threehump = sample_data(bb_cls, num_train, num_grid,
        #    expand_datadct=False)
        #print("Storing files to " + trg_path)
        #os.makedirs(trg_path, exist_ok=True)
        #fname = os.path.join(
        #    trg_path, fname_msk.format(data_name, num_train, 'data'))
        #with open(fname, 'wb') as handle:
        #    pickle.dump(threehump, handle, protocol=pickle.HIGHEST_PROTOCOL)
        #threehump = expand_data(threehump)
        #
        #threehump_gpy = fit_predict_gpy(
        #    threehump, random_seed, optimization_config)
        #fname = os.path.join(
        #    trg_path, fname_msk.format(data_name, num_train, 'gpy'))
        #with open(fname, 'wb') as handle:
        #    pickle.dump(threehump_gpy, handle, protocol=pickle.HIGHEST_PROTOCOL)
        # --------------------------------------------------------------------

        # test_intermediates = dict()  # DEBUG
        test_intermediates = None
        threehump_ours = fit_predict_ours(
            threehump, random_seed, optimization_config,
            test_intermediates=test_intermediates)
        # DEBUG:
        # compare_gpy_predict_posterior_marginals(
        #     test_intermediates, noise_variance_gpy=None)
        # END DEBUG

        # If you want a visual result, uncomment the following lines
        # title = "ThreeHump, num_train={}".format(num_train)
        # plot_predictions(threehump, threehump_ours, threehump_gpy, title)

        # ThreeHump - means
        sse = sum((threehump_gpy['means'].reshape(-1) - threehump_ours['means'])**2)
        N = threehump_ours['means'].shape[0]
        threehump_means_rmse = np.sqrt(sse / N)
        assert threehump_means_rmse <= 5e-2
        # ThreeHump - stds
        sse = sum((threehump_gpy['stddevs'].reshape(-1) - threehump_ours['stddevs'])**2)
        N = threehump_ours['stddevs'].shape[0]
        threehump_stds_rmse = np.sqrt(sse / N)
        assert threehump_stds_rmse <= 6e-2

    if do_ackley:
        random_seed = 232098764
        data_name = 'ackley'
        fname = fname_msk.format(data_name, num_train, 'data')
        ackley = download_pickle_file(fname)
        fname = fname_msk.format(data_name, num_train, 'gpy')
        ackley_gpy = download_pickle_file(fname)
        # --------------------------------------------------------------------
        #bb_cls = Ackley
        #ackley = sample_data(bb_cls, num_train, num_grid,
        #    expand_datadct=False)
        #print("Storing files to " + trg_path)
        #os.makedirs(trg_path, exist_ok=True)
        #fname = os.path.join(
        #    trg_path, fname_msk.format(data_name, num_train, 'data'))
        #with open(fname, 'wb') as handle:
        #    pickle.dump(ackley, handle, protocol=pickle.HIGHEST_PROTOCOL)
        #ackley = expand_data(ackley)
        #
        #ackley_gpy = fit_predict_gpy(ackley, random_seed, optimization_config)
        #fname = os.path.join(
        #    trg_path, fname_msk.format(data_name, num_train, 'gpy'))
        #with open(fname, 'wb') as handle:
        #    pickle.dump(ackley_gpy, handle, protocol=pickle.HIGHEST_PROTOCOL)
        # --------------------------------------------------------------------

        # test_intermediates = dict()  # DEBUG
        test_intermediates = None
        ackley_ours = fit_predict_ours(
            ackley, random_seed, optimization_config,
            test_intermediates=test_intermediates)
        # DEBUG:
        # compare_gpy_predict_posterior_marginals(
        #     test_intermediates, noise_variance_gpy=None)
        # END DEBUG

        # If you want a visual result, uncomment the following lines
        # title = "Ackley, num_train={}".format(num_train)
        # plot_predictions(ackley, ackley_ours, ackley_gpy, title)

        # Ackley - means
        sse = sum((ackley_gpy['means'].reshape(-1) - ackley_ours['means'])**2)
        N = ackley_ours['means'].shape[0]
        ackley_means_rmse = np.sqrt(sse / N)
        # print('ackley_means = {}'.format(ackley_means_rmse))
        assert ackley_means_rmse <= 7e-3
        # Ackley - stds
        sse = sum((ackley_gpy['stddevs'].reshape(-1) - ackley_ours['stddevs'])**2)
        N = ackley_ours['stddevs'].shape[0]
        ackley_stds_rmse = np.sqrt(sse / N)
        # print('ackley_stds = {}'.format(ackley_stds_rmse))
        assert ackley_stds_rmse <= 3e-3


if __name__ == "__main__":
    test_comparison_gpy()
