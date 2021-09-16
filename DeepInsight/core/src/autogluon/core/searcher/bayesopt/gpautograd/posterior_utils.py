import autograd.numpy as anp
import autograd.scipy.linalg as aspl
from autograd.builtins import isinstance
from autograd.tracer import getval

from .constants import \
    NOISE_VARIANCE_LOWER_BOUND, MIN_POSTERIOR_VARIANCE, \
    MIN_CHOLESKY_DIAGONAL_VALUE
from .custom_op import AddJitterOp, cholesky_factorization, flatten_and_concat


def cholesky_computations(
        features, targets, mean, kernel, noise_variance, debug_log=False,
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

    :param features: Input matrix X (n,d)
    :param targets: Target matrix Y (n,m)
    :param mean: Mean function
    :param kernel: Kernel function
    :param noise_variance: Noise variance (may be increased)
    :param debug_log: Debug output during add_jitter CustomOp?
    :param test_intermediates: If given, all intermediates are written into
        this dict
    :return: L, P
    """
    kernel_mat = kernel(features, features)
#   Add jitter to noise_variance (if needed) in order to guarantee that
#   Cholesky factorization works
    sys_mat = AddJitterOp(
        flatten_and_concat(kernel_mat, noise_variance),
        initial_jitter_factor=NOISE_VARIANCE_LOWER_BOUND,
        debug_log='true' if debug_log else 'false')
    chol_fact = cholesky_factorization(sys_mat)
    centered_y = targets - anp.reshape(mean(features), (-1, 1))
    pred_mat = aspl.solve_triangular(chol_fact, centered_y, lower=True)
    # print('chol_fact', chol_fact)

    if test_intermediates is not None:
        assert isinstance(test_intermediates, dict)
        test_intermediates.update({
            'features': features,
            'targets': targets,
            'noise_variance': noise_variance,
            'kernel_mat': kernel_mat,
            'sys_mat': sys_mat,
            'chol_fact': chol_fact,
            'pred_mat': pred_mat,
            'centered_y': centered_y})
        test_intermediates.update(kernel.get_params())
        test_intermediates.update(mean.get_params())   
    
    return chol_fact, pred_mat


def predict_posterior_marginals(
        features, mean, kernel, chol_fact, pred_mat, test_features,
        test_intermediates=None):
    """
    Computes posterior means and variances for test_features.
    If pred_mat is a matrix, so will be posterior_means, but not
    posterior_variances. Reflects the fact that for GP regression and fixed
    hyperparameters, the posterior mean depends on the targets y, but the
    posterior covariance does not.

    :param features: Training inputs
    :param mean: Mean function
    :param kernel: Kernel function
    :param chol_fact: Part L of posterior state
    :param pred_mat: Part P of posterior state
    :param test_features: Test inputs
    :return: posterior_means, posterior_variances
    """
    k_tr_te = kernel(features, test_features)
    linv_k_tr_te = aspl.solve_triangular(chol_fact, k_tr_te, lower=True)
    posterior_means = anp.matmul(anp.transpose(linv_k_tr_te), pred_mat) + \
                      anp.reshape(mean(test_features), (-1, 1))
    posterior_variances = kernel.diagonal(test_features) - anp.sum(
        anp.square(linv_k_tr_te), axis=0)
    if test_intermediates is not None:
        assert isinstance(test_intermediates, dict)
        test_intermediates.update({
            'k_tr_te': k_tr_te,
            'linv_k_tr_te': linv_k_tr_te,
            'test_features': test_features,
            'pred_means': posterior_means,
            'pred_vars': anp.reshape(anp.maximum(
                posterior_variances, MIN_POSTERIOR_VARIANCE),
                (-1,))})    

    return posterior_means, anp.reshape(anp.maximum(
        posterior_variances, MIN_POSTERIOR_VARIANCE), (-1,))


def sample_posterior_marginals(
        features, mean, kernel, chol_fact, pred_mat, test_features,
        num_samples=1):
    """
    Draws num_sample samples from the product of marginals of the posterior
    over input points test_features. If pred_mat is a matrix with m columns,
    the samples returned have shape (n_test, m, num_samples).

    :param features: Training inputs
    :param mean: Mean function
    :param kernel: Kernel function
    :param chol_fact: Part L of posterior state
    :param pred_mat: Part P of posterior state
    :param test_features: Test inputs
    :param num_samples: Number of samples to draw
    :return: Samples, shape (n_test, num_samples) or (n_test, m, num_samples)
    """
    post_means, post_vars = predict_posterior_marginals(
        features, mean, kernel, chol_fact, pred_mat, test_features)
    post_means = anp.expand_dims(post_means, axis=-1)  # (n_test, m, 1)
    post_stds = anp.sqrt(anp.reshape(post_vars, (-1, 1, 1))) # (n_test, 1, 1)
    n01_vecs = [anp.random.normal(size=getval(post_means.shape))
                for _ in range(num_samples)]
    n01_mat = anp.concatenate(n01_vecs, axis=-1)
    samples = anp.multiply(n01_mat, post_stds) + post_means
    
    if samples.shape[1] == 1:
        n_test = getval(samples.shape)[0]
        samples = anp.reshape(samples, (n_test, -1))  # (n_test, num_samples)
        
    return samples


def sample_posterior_joint(
        features, mean, kernel, chol_fact, pred_mat, test_features,
        num_samples=1):
    """
    Draws num_sample samples from joint posterior distribution over inputs
    test_features. This is done by computing mean and covariance matrix of
    this posterior, and using the Cholesky decomposition of the latter. If
    pred_mat is a matrix with m columns, the samples returned have shape
    (n_test, m, num_samples).

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
    linv_k_tr_te = aspl.solve_triangular(chol_fact, k_tr_te, lower=True)
    posterior_mean = anp.matmul(anp.transpose(linv_k_tr_te), pred_mat) + \
                     anp.reshape(mean(test_features), (-1, 1))
    posterior_cov = kernel(test_features, test_features) - anp.dot(
        anp.transpose(linv_k_tr_te), linv_k_tr_te)
    jitter_init = anp.ones((1,)) * (1e-5)
    sys_mat = AddJitterOp(
        flatten_and_concat(posterior_cov, jitter_init),
        initial_jitter_factor=NOISE_VARIANCE_LOWER_BOUND)
    lfact = cholesky_factorization(sys_mat)
    # Draw samples
    # posterior_mean.shape = (n_test, m), where m is number of cols of pred_mat
    # Reshape to (n_test, m, 1)
    n_test = getval(posterior_mean.shape)[0]
    posterior_mean = anp.expand_dims(posterior_mean, axis=-1)
    n01_vecs = [anp.random.normal(size=getval(posterior_mean.shape))
                for _ in range(num_samples)]
    n01_mat = anp.reshape(anp.concatenate(n01_vecs, axis=-1), (n_test, -1))
    samples = anp.reshape(
        anp.dot(lfact, n01_mat), (n_test, -1, num_samples))
    samples = samples + posterior_mean
    
    if samples.shape[1] == 1:
        samples = anp.reshape(samples, (n_test, -1))  # (n_test, num_samples)
        
    return samples


def _compute_lvec(features, chol_fact, kernel, feature):
    kvec = anp.reshape(kernel(features, feature), (-1, 1))
    return anp.reshape(aspl.solve_triangular(
        chol_fact, kvec, lower=True), (1, -1))


def cholesky_update(
        features, chol_fact, pred_mat, mean, kernel, noise_variance,
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
        lvec = _compute_lvec(features, chol_fact, kernel, feature)
    kscal = anp.reshape(kernel.diagonal(feature), (1,))
    noise_variance = anp.reshape(noise_variance, (1,))
    lsqscal = anp.maximum(
        kscal + noise_variance - anp.sum(anp.square(lvec)),
        MIN_CHOLESKY_DIAGONAL_VALUE ** 2)
    lscal = anp.reshape(anp.sqrt(lsqscal), (1, 1))
    mscal = anp.reshape(mean(feature), (1, 1))
    pvec = target - mscal
    pvec = anp.divide(pvec - anp.matmul(lvec, pred_mat), lscal)
    pred_mat_new = anp.concatenate([pred_mat, pvec], axis=0)
    zerovec = anp.zeros((getval(lvec.size), 1))
    chol_fact_new = anp.concatenate([
        anp.concatenate([chol_fact, lvec], axis=0),
        anp.concatenate([zerovec, lscal], axis=0)], axis=1)

    return chol_fact_new, pred_mat_new

# Specialized routine, used in IncrementalUpdateGPPosteriorState.
# The idea is to share the computation of lvec between sampling a new target
# value and incremental Cholesky update.
# If mean_impute_mask is given, it is a boolean vector of size m (number
# columns of pred_mat). Columns j of target, where mean_impute_ mask[j] is
# true, are set to the predictive mean (instead of being sampled).
def sample_and_cholesky_update(
        features, chol_fact, pred_mat, mean, kernel, noise_variance,
        feature, mean_impute_mask=None):
    # Draw sample target. Also, lvec is reused below
    lvec = _compute_lvec(features, chol_fact, kernel, feature)
    pred_mean = anp.dot(lvec, pred_mat) + anp.reshape(mean(feature), (1, 1))
    # Note: We do not add noise_variance to the predictive variance
    pred_std = anp.reshape(anp.sqrt(anp.maximum(
        kernel.diagonal(feature) - anp.sum(anp.square(lvec)),
        MIN_POSTERIOR_VARIANCE)), (1, 1))
    n01mat = anp.random.normal(size=getval(pred_mean.shape))
    if mean_impute_mask is not None:
        assert len(mean_impute_mask) == pred_mat.shape[1]
        n01mat[0, mean_impute_mask] = 0
    target = pred_mean + anp.multiply(n01mat, pred_std)
    chol_fact_new, pred_mat_new = cholesky_update(
        features, chol_fact, pred_mat, mean, kernel, noise_variance,
        feature, target, lvec=lvec)
    features_new = anp.concatenate([features, feature], axis=0)
    
    return chol_fact_new, pred_mat_new, features_new, target


def negative_log_marginal_likelihood(chol_fact, pred_mat):
    """
    The marginal likelihood is only computed if pred_mat has a single column
    (not for fantasy sample case).
    """
    assert pred_mat.ndim == 1 or pred_mat.shape[1] == 1,\
        "Multiple target vectors are not supported"
    sqnorm_predmat = anp.sum(anp.square(pred_mat))
    logdet_cholfact = 2.0 * anp.sum(anp.log(anp.abs(anp.diag(chol_fact))))
    n_samples = getval(pred_mat.size)

    return 0.5 * (sqnorm_predmat + n_samples * anp.log(2 * anp.pi) + logdet_cholfact)
