import numpy as np
import autograd.numpy as anp
from autograd import grad
from autograd.tracer import getval
from typing import Tuple, Optional, Dict

from .kernel import KernelFunction
from .mean import MeanFunction
from .posterior_utils import cholesky_computations, predict_posterior_marginals,\
    sample_posterior_marginals, sample_posterior_joint, cholesky_update,\
    negative_log_marginal_likelihood, sample_and_cholesky_update


class GaussProcPosteriorState(object):
    """
    Represent posterior state for Gaussian process regression model.
    Note that members are immutable. If the posterior state is to be
    updated, a new object is created and returned.
    """
    def __init__(
            self, features: np.ndarray, targets: Optional[np.ndarray],
            mean: MeanFunction, kernel: KernelFunction,
            noise_variance: np.ndarray, debug_log: bool = False,
            test_intermediates: Optional[dict] = None,
            **kwargs):
        """
        If targets has m > 1 columns, they correspond to fantasy samples.

        If targets is None, this is an internal (copy) constructor, where
        kwargs contains chol_fact, pred_mat.

        :param features: Input points X, shape (n, d)
        :param targets: Targets Y, shape (n, m)
        :param mean: Mean function m(X)
        :param kernel: Kernel function k(X, X')
        :param noise_variance: Noise variance sigsq, shape (1,)
        :param test_intermediates: See cholesky_computations
        """
        self.mean = mean
        self.kernel = kernel
        self.noise_variance = anp.array(noise_variance, copy=True)
        if targets is not None:
            targets_shape = getval(targets.shape)
            targets = anp.reshape(targets, (targets_shape[0], -1))

            chol_fact, pred_mat = cholesky_computations(
                features, targets, mean, kernel, noise_variance,
                debug_log=debug_log, test_intermediates=test_intermediates)
            
            self.features = anp.array(features, copy=True)
            self.chol_fact = chol_fact
            self.pred_mat = pred_mat
            self._test_intermediates = test_intermediates
        else:
            # Internal (copy) constructor
            self.features = features
            self.chol_fact = kwargs['chol_fact']
            self.pred_mat = kwargs['pred_mat']

    @property
    def num_data(self):
        return self.features.shape[0]

    @property
    def num_features(self):
        return self.features.shape[1]

    @property
    def num_fantasies(self):
        return self.pred_mat.shape[1]

    def _state_args(self):
        return [self.features, self.mean, self.kernel, self.chol_fact,
            self.pred_mat]

    def predict(self, test_features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return predict_posterior_marginals(
            *self._state_args(), test_features,
            test_intermediates=self._test_intermediates)

    def sample_joint(self, test_features: np.ndarray, num_samples: int=1) \
            -> np.ndarray:
        return sample_posterior_joint(
            *self._state_args(), test_features, num_samples)

    def sample_marginals(self, test_features: np.ndarray, num_samples: int=1) \
            -> np.ndarray:
        return sample_posterior_marginals(
            *self._state_args(), test_features, num_samples)

    def neg_log_likelihood(self) -> anp.ndarray:
        """
        Works only if fantasy samples are not used (single targets vector).

        :return: Negative log (marginal) likelihood
        """
        return negative_log_marginal_likelihood(
            self.chol_fact, self.pred_mat)

    def backward_gradient(
            self, input: np.ndarray,
            head_gradients: Dict[str, np.ndarray],
            mean_data: float, std_data: float) -> np.ndarray:
        """
        Implements SurrogateModel.backward_gradient, see comments there.
        This is for a single posterior state. If the SurrogateModel uses
        MCMC, have to call this for every sample.

        The posterior represented here is based on normalized data, while
        the acquisition function is based on the de-normalized predictive
        distribution, which is why we need 'mean_data', 'std_data' here.

        :param input: Single input point x, shape (d,)
        :param head_gradients: See SurrogateModel.backward_gradient
        :param mean_data: Mean used to normalize targets
        :param std_data: Stddev used to normalize targets
        :return:
        """
        test_feature = np.reshape(input, (1, -1))
        # Compute de-normalized mean, std and
        # backward with specific head gradients
        def diff_test_feature(test_feature_array):
            norm_mean, norm_variance = self.predict(test_feature_array)
            # De-normalize, and variance -> stddev
            pred_mean = norm_mean * std_data + mean_data
            pred_std = anp.sqrt(norm_variance) * std_data
            head_gradients_mean = anp.reshape(head_gradients['mean'], pred_mean.shape)
            head_gradients_std = anp.reshape(head_gradients['std'], pred_std.shape)
            # Added to mimic mxnet.autograd.backward
            pred_mean_sum = anp.sum(anp.multiply(pred_mean, head_gradients_mean))
            pred_std_sum = anp.sum(anp.multiply(pred_std, head_gradients_std))
            return pred_mean_sum + pred_std_sum
        
        test_feature_gradient = grad(diff_test_feature)
        
        return np.reshape(test_feature_gradient(test_feature), input.shape)


class IncrementalUpdateGPPosteriorState(GaussProcPosteriorState):
    """
    Extension of GaussProcPosteriorState which allows for incremental
    updating, given that a single data case is appended to the training
    set.

    In order to not mutate members, 
    "the update method returns a new object."
    """
    def __init__(
            self, features: np.ndarray, targets: Optional[np.ndarray],
            mean: MeanFunction, kernel: KernelFunction,
            noise_variance: np.ndarray, **kwargs):
        
        super(IncrementalUpdateGPPosteriorState, self).__init__(
            features, targets, mean, kernel, noise_variance, **kwargs)

    def update(self, feature: np.ndarray, target: np.ndarray) -> 'IncrementalUpdateGPPosteriorState':
        """
        :param feature: Additional input xstar, shape (1, d)
        :param target: Additional target ystar, shape (1, m)
        :return: Posterior state for increased data set
        """
        feature = anp.reshape(feature, (1, -1))
        target = anp.reshape(target, (1, -1))
        assert feature.shape[1] == self.features.shape[1], \
            "feature.shape[1] = {} != {} = self.features.shape[1]".format(
                feature.shape[1], self.features.shape[1])
        assert target.shape[1] == self.pred_mat.shape[1], \
            "target.shape[1] = {} != {} = self.pred_mat.shape[1]".format(
                target.shape[1], self.pred_mat.shape[1])
        chol_fact_new, pred_mat_new = cholesky_update(
            self.features, self.chol_fact, self.pred_mat, self.mean,
            self.kernel, self.noise_variance, feature, target)
        features_new = anp.concatenate([self.features, feature], axis=0)
        state_new = IncrementalUpdateGPPosteriorState(
            features=features_new,
            targets=None,
            mean=self.mean,
            kernel=self.kernel,
            noise_variance=self.noise_variance,
            chol_fact=chol_fact_new,
            pred_mat=pred_mat_new)
        return state_new

    def sample_and_update(self, feature: np.ndarray, mean_impute_mask=None) -> \
        (np.ndarray, 'IncrementalUpdateGPPosteriorState'):
        """
        Draw target(s), shape (1, m), from current posterior state, then update
        state based on these. The main computation of lvec is shared among the
        two.
        If mean_impute_mask is given, it is a boolean vector of size m (number
        columns of pred_mat). Columns j of target, where mean_impute_ mask[j]
        is true, are set to the predictive mean (instead of being sampled).

        :param feature: Additional input xstar, shape (1, d)
        :param mean_impute_mask: See above
        :return: target, poster_state_new
        """
        feature = anp.reshape(feature, (1, -1))
        assert feature.shape[1] == self.features.shape[1], \
            "feature.shape[1] = {} != {} = self.features.shape[1]".format(
                feature.shape[1], self.features.shape[1])
        chol_fact_new, pred_mat_new, features_new, target = \
            sample_and_cholesky_update(
                self.features, self.chol_fact, self.pred_mat, self.mean,
                self.kernel, self.noise_variance, feature, mean_impute_mask)
        state_new = IncrementalUpdateGPPosteriorState(
            features=features_new,
            targets=None,
            mean=self.mean,
            kernel=self.kernel,
            noise_variance=self.noise_variance,
            chol_fact=chol_fact_new,
            pred_mat=pred_mat_new)
        return target, state_new

    def expand_fantasies(self, num_fantasies: int) \
            -> 'IncrementalUpdateGPPosteriorState':
        """
        If this posterior has been created with a single targets vector,
        shape (n, 1), use this to duplicate this vector m = num_fantasies
        times. Call this method before fantasy targets are appended by
        update.

        :param num_fantasies: Number m of fantasy samples
        :return: New state with targets duplicated m times
        """
        assert num_fantasies > 1
        assert self.pred_mat.shape[1] == 1, \
            "Method requires posterior without fantasy samples"
        pred_mat_new = anp.concatenate(
            ([self.pred_mat] * num_fantasies), axis=1)
        state_new = IncrementalUpdateGPPosteriorState(
            features=self.features,
            targets=None,
            mean=self.mean,
            kernel=self.kernel,
            noise_variance=self.noise_variance,
            chol_fact=self.chol_fact,
            pred_mat=pred_mat_new)
        
        return state_new
