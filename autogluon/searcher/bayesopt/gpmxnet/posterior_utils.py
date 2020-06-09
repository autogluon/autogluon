from typing import Union
import mxnet as mx
import numpy as np

from autogluon.searcher.bayesopt.gpmxnet.constants import \
    NOISE_VARIANCE_LOWER_BOUND, MIN_POSTERIOR_VARIANCE, \
    MIN_CHOLESKY_DIAGONAL_VALUE
from autogluon.searcher.bayesopt.gpmxnet.custom_op import AddJitterOp, \
    AddJitterOpProp


Tensor = Union[mx.nd.NDArray, mx.sym.Symbol]


def mxnet_F(x: Tensor):
    if isinstance(x, mx.nd.NDArray):
        return mx.nd
    else:
        return mx.sym


def mxnet_is_ndarray(F):
    return F.__name__ == 'mxnet.ndarray'


def cholesky_computations(
        F, features, targets, mean, kernel, noise_variance, debug_log=False,
        test_intermediates=None):
    """
    Given input matrix X (features), target matrix Y (targets), mean and kernel
    function, compute posterior state {L, P}, where L is the Cholesky factor
    of
        k(X, X) + sigsq_final * I
    and
        L P = Y - mean(X)
    Here, sigsq_final >= noise_variance is minimal such that the Cholesky
    factorization does not fail.

    :param F: mx.nd or mx.sym
    :param features: Input matrix X
    :param targets: Target matrix Y
    :param mean: Mean function
    :param kernel: Kernel function
    :param noise_variance: Noise variance (may be increased)
    :param debug_log: Debug output during add_jitter CustomOp?
    :param test_intermediates: If given, all intermediates are written into
        this dict (only if F = mx.nd)
    :return: L, P

    """
    kernel_mat = kernel(features, features)
    # Add jitter to noise_variance (if needed) in order to guarantee that
    # Cholesky factorization works
    sys_mat = F.Custom(
        kernel_mat, noise_variance, name="add_jitter", op_type='add_jitter',
        initial_jitter_factor=NOISE_VARIANCE_LOWER_BOUND,
        debug_log='true' if debug_log else 'false')
    chol_fact = F.linalg.potrf(sys_mat)
    centered_y = F.broadcast_sub(
        targets, F.reshape(mean(features), shape=(-1, 1)))
    pred_mat = F.linalg.trsm(chol_fact, centered_y)
    # For testing:
    if test_intermediates is not None and mxnet_is_ndarray(F):
        assert isinstance(test_intermediates, dict)
        test_intermediates.update({
            'features': features.asnumpy(),
            'targets': targets.asnumpy(),
            'noise_variance': noise_variance.asscalar(),
            'kernel_mat': kernel_mat.asnumpy(),
            'sys_mat': sys_mat.asnumpy(),
            'chol_fact': chol_fact.asnumpy(),
            'pred_mat': pred_mat.asnumpy(),
            'centered_y': centered_y.asnumpy()})
        test_intermediates.update(kernel.get_params())
        test_intermediates.update(mean.get_params())
    return chol_fact, pred_mat


def predict_posterior_marginals(
        F, features, mean, kernel, chol_fact, pred_mat, test_features,
        test_intermediates=None):
    """
    Computes posterior means and variances for test_features.
    If pred_mat is a matrix, so will be posterior_means, but not
    posterior_variances. Reflects the fact that for GP regression and fixed
    hyperparameters, the posterior mean depends on the targets y, but the
    posterior covariance does not.

    :param F: mx.sym or mx.nd
    :param features: Training inputs
    :param mean: Mean function
    :param kernel: Kernel function
    :param chol_fact: Part L of posterior state
    :param pred_mat: Part P of posterior state
    :param test_features: Test inputs
    :return: posterior_means, posterior_variances

    """
    k_tr_te = kernel(features, test_features)
    linv_k_tr_te = F.linalg.trsm(chol_fact, k_tr_te)
    posterior_means = F.broadcast_add(
        F.linalg.gemm2(
            linv_k_tr_te, pred_mat, transpose_a=True, transpose_b=False),
        F.reshape(mean(test_features), shape=(-1, 1)))
    posterior_variances = kernel.diagonal(F, test_features) - F.sum(
        F.square(linv_k_tr_te), axis=0)
    # For testing:
    if test_intermediates is not None and mxnet_is_ndarray(F):
        assert isinstance(test_intermediates, dict)
        test_intermediates.update({
            'k_tr_te': k_tr_te.asnumpy(),
            'linv_k_tr_te': linv_k_tr_te.asnumpy(),
            'test_features': test_features.asnumpy(),
            'pred_means': posterior_means.asnumpy(),
            'pred_vars': F.reshape(F.maximum(
                posterior_variances, MIN_POSTERIOR_VARIANCE),
                shape=(-1,)).asnumpy()})
    return posterior_means, F.reshape(F.maximum(
        posterior_variances, MIN_POSTERIOR_VARIANCE), shape=(-1,))


def sample_posterior_marginals(
        F, features, mean, kernel, chol_fact, pred_mat, test_features,
        num_samples=1):
    """
    Draws num_sample samples from the product of marginals of the posterior
    over input points test_features. If pred_mat is a matrix with m columns,
    the samples returned have shape (n_test, m, num_samples).

    Note: If F = mx.sym, the samples always have shape
    (n_test, m, num_samples), even if m = 1.

    :param F: mx.sym or mx.nd
    :param features: Training inputs
    :param mean: Mean function
    :param kernel: Kernel function
    :param chol_fact: Part L of posterior state
    :param pred_mat: Part P of posterior state
    :param test_features: Test inputs
    :param num_samples: Number of samples to draw
    :return: Samples, shape (n_test, num_samples) or (n_test, m, num_samples)

    """
    # Shape of post_means is (n_test, m)
    post_means, post_vars = predict_posterior_marginals(
        F, features, mean, kernel, chol_fact, pred_mat, test_features)
    post_means = F.expand_dims(post_means, axis=-1)  # (n_test, m, 1)
    post_stds = F.sqrt(F.reshape(post_vars, shape=(-1, 1, 1)))  # (n_test, 1, 1)
    n01_vecs = [F.random.normal_like(post_means) for _ in range(num_samples)]
    n01_mat = F.concat(*n01_vecs, dim=-1)  # (n_test, m, num_samples)
    samples = F.broadcast_add(F.broadcast_mul(n01_mat, post_stds), post_means)
    # Remove m == 1 dimension. Can be done only for mx.nd
    if mxnet_is_ndarray(F) and samples.shape[1] == 1:
        samples = F.reshape(samples, shape=(0, -1))  # (n_test, num_samples)
    return samples


def sample_posterior_joint(
        F, features, mean, kernel, chol_fact, pred_mat, test_features,
        num_samples=1):
    """
    Draws num_sample samples from joint posterior distribution over inputs
    test_features. This is done by computing mean and covariance matrix of
    this posterior, and using the Cholesky decomposition of the latter. If
    pred_mat is a matrix with m columns, the samples returned have shape
    (n_test, m, num_samples).

    Note: If F = mx.sym, the samples always have shape
    (n_test, m, num_samples), even if m = 1.

    :param F: mx.sym or mx.nd
    :param features: Training inputs
    :param mean: Mean function
    :param kernel: Kernel function
    :param chol_fact: Part L of posterior state
    :param pred_mat: Part P of posterior state
    :param test_features: Test inputs
    :param num_samples: Number of samples to draw
    :return: Samples, shape (n_test, num_samples) or (n_test, m, num_samples)

    """
    k_tr_te = kernel(features, test_features)
    linv_k_tr_te = F.linalg.trsm(chol_fact, k_tr_te)
    posterior_mean = F.broadcast_add(
        F.linalg.gemm2(
            linv_k_tr_te, pred_mat, transpose_a=True, transpose_b=False),
        F.reshape(mean(test_features), shape=(-1, 1)))
    posterior_cov = kernel(test_features, test_features) - F.linalg.syrk(
        linv_k_tr_te, transpose=True)
    # Compute the Cholesky decomposition of posterior_cov (expensive!)
    # Add some jitter proactively (>= 1e-5)
    ones_like_test = F.ones_like(F.reshape(F.slice_axis(
        F.BlockGrad(test_features), axis=1, begin=0, end=1), shape=(-1,)))
    jitter_init = F.ones_like(
        F.slice_axis(ones_like_test, axis=0, begin=0, end=1)) * (1e-5)
    sys_mat = F.Custom(
        posterior_cov, jitter_init, name="add_jitter",
        op_type='add_jitter', initial_jitter_factor=NOISE_VARIANCE_LOWER_BOUND)
    lfact = F.linalg.potrf(sys_mat)
    # Draw samples
    # posterior_mean.shape = (n_test, m), where m is number of cols of pred_mat
    # Reshape to (n_test, m, 1)
    posterior_mean = F.expand_dims(posterior_mean, axis=-1)
    n01_vecs = [
        F.random.normal_like(posterior_mean) for _ in range(num_samples)]
    # n01_mat.shape = (n_test, m, num_samples) -> (n_test, m * num_samples)
    n01_mat = F.reshape(F.concat(*n01_vecs, dim=-1), shape=(0, -1))
    # Reshape samples back to (n_test, m, num_samples) after trmm
    samples = F.reshape(
        F.linalg.trmm(lfact, n01_mat), shape=(0, -1, num_samples))
    samples = F.broadcast_add(samples, posterior_mean)
    # Remove m == 1 dimension. Can be done only for mx.nd
    if mxnet_is_ndarray(F) and samples.shape[1] == 1:
        samples = F.reshape(samples, shape=(0, -1))  # (n_test, num_samples)
    return samples


def _compute_lvec(F, features, chol_fact, kernel, feature):
    kvec = F.reshape(kernel(features, feature), shape=(-1, 1))
    return F.reshape(F.linalg.trsm(chol_fact, kvec), shape=(1, -1))


def cholesky_update(
        F, features, chol_fact, pred_mat, mean, kernel, noise_variance,
        feature, target, lvec=None):
    """
    Incremental update of posterior state (Cholesky factor, prediction
    matrix), given one datapoint (feature, target).

    Note: noise_variance is the initial value, before any jitter may have
    been added to compute chol_fact. Here, we add the minimum amount of
    jitter such that the new diagonal entry of the Cholesky factor is
    >= MIN_CHOLESKY_DIAGONAL_VALUE. This means that if cholesky_update is
    used several times, we in fact add a diagonal (but not spherical)
    jitter matrix.

    :param F:
    :param features: Shape (n, d)
    :param chol_fact: Shape (n, n)
    :param pred_mat: Shape (n, m)
    :param mean:
    :param kernel:
    :param noise_variance:
    :param feature: Shape (1, d)
    :param target: Shape (1, m)
    :param lvec: If given, this is the new column of the Cholesky factor
        except the diagonal entry. If not, this is computed here
    :return: chol_fact_new (n+1, n+1), pred_mat_new (n+1, m)

    """
    if lvec is None:
        lvec = _compute_lvec(F, features, chol_fact, kernel, feature)
    kscal = F.reshape(kernel.diagonal(F, feature), shape=(1,))
    noise_variance = F.reshape(noise_variance, shape=(1,))
    lsqscal = F.maximum(kscal + noise_variance - F.sum(F.square(lvec)),
                        MIN_CHOLESKY_DIAGONAL_VALUE ** 2)
    lscal = F.reshape(F.sqrt(lsqscal), shape=(1, 1))
    mscal = F.reshape(mean(feature), shape=(1, 1))
    pvec = F.broadcast_sub(target, mscal)
    pvec = F.broadcast_div(pvec - F.linalg.gemm2(lvec, pred_mat), lscal)
    pred_mat_new = F.concat(pred_mat, pvec, dim=0)
    tmpmat = F.concat(chol_fact, lvec, dim=0)
    zerovec = F.zeros_like(F.reshape(lvec, shape=(-1, 1)))
    chol_fact_new = F.concat(
        tmpmat, F.concat(zerovec, lscal, dim=0), dim=1)
    return chol_fact_new, pred_mat_new


# Specialized routine, used in GPPosteriorStateIncrementalUpdater.
# The idea is to share the computation of lvec between sampling a new target
# value and incremental Cholesky update
def sample_and_cholesky_update(
        F, features, chol_fact, pred_mat, mean, kernel, noise_variance,
        feature):
    # Draw sample target. Also, lvec is reused below
    lvec = _compute_lvec(F, features, chol_fact, kernel, feature)
    pred_mean = F.broadcast_add(
        F.dot(lvec, pred_mat), F.reshape(mean(feature), shape=(1, 1)))
    # Note: We do not add noise_variance to the predictive variance
    pred_std = F.reshape(F.sqrt(F.maximum(
        kernel.diagonal(F, feature) - F.sum(F.square(lvec)),
        MIN_POSTERIOR_VARIANCE)), shape=(1, 1))
    n01mat = F.random.normal_like(pred_mean)
    target = pred_mean + F.broadcast_mul(n01mat, pred_std)
    # Incremental posterior update
    chol_fact_new, pred_mat_new = cholesky_update(
        F, features, chol_fact, pred_mat, mean, kernel, noise_variance,
        feature, target, lvec=lvec)
    features_new = F.concat(features, feature, dim=0)
    return chol_fact_new, pred_mat_new, features_new, target


def negative_log_marginal_likelihood(F, chol_fact, pred_mat):
    """
    The marginal likelihood is only computed if pred_mat has a single column
    (not for fantasy sample case).
    """

    if mxnet_is_ndarray(F):
        assert pred_mat.ndim == 1 or pred_mat.shape[1] == 1, \
            "Multiple target vectors are not supported"
    sqnorm_predmat = F.sum(F.square(pred_mat))
    logdet_cholfact = 2.0 * F.sum(F.log(F.abs(F.diag(chol_fact))))
    # pred_mat is a vector of dimension the number of samples
    n_samples = F.sum(F.ones_like(F.BlockGrad(pred_mat)))
    return 0.5 * (sqnorm_predmat + n_samples * np.log(2 * np.pi) + logdet_cholfact)
