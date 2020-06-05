from typing import Tuple, Optional
import mxnet as mx

from autogluon.searcher.bayesopt.gpmxnet.kernel import KernelFunction
from autogluon.searcher.bayesopt.gpmxnet.mean import MeanFunction
from autogluon.searcher.bayesopt.gpmxnet.posterior_utils import Tensor, \
    mxnet_F, cholesky_computations, predict_posterior_marginals, \
    sample_posterior_marginals, sample_posterior_joint, cholesky_update, \
    negative_log_marginal_likelihood, mxnet_is_ndarray


class GaussProcPosteriorState(object):
    """
    Represent posterior state for Gaussian process regression model.
    Note that members are immutable. If the posterior state is to be
    updated, a new object is created and returned.

    """
    def __init__(
            self, features: Tensor, targets: Optional[Tensor],
            mean: MeanFunction, kernel: KernelFunction,
            noise_variance: Tensor, debug_log: bool = False,
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
        F = mxnet_F(features)
        self.F = F
        self.mean = mean
        self.kernel = kernel
        if targets is not None:
            targets = F.reshape(targets, shape=(0, -1))
            chol_fact, pred_mat = cholesky_computations(
                F, features, targets, mean, kernel, noise_variance,
                debug_log=debug_log, test_intermediates=test_intermediates)
            if self.is_ndarray():
                # Make sure the computations are done, and not deferred until
                # later. This is important in order to get profiling right
                # (and there is no advantage in deferring at this point)
                chol_fact.wait_to_read()
                pred_mat.wait_to_read()
            # Make copy, just to be safe:
            self.features = features if F == mx.sym else features.copy()
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
        self._check_is_ndarray()
        return self.features.shape[0]

    @property
    def num_features(self):
        self._check_is_ndarray()
        return self.features.shape[1]

    @property
    def num_fantasies(self):
        self._check_is_ndarray()
        return self.pred_mat.shape[1]

    def _state_args(self):
        return [self.F, self.features, self.mean, self.kernel, self.chol_fact,
            self.pred_mat]

    def predict(self, test_features: Tensor) -> Tuple[Tensor, Tensor]:
        return predict_posterior_marginals(
            *self._state_args(), test_features,
            test_intermediates=self._test_intermediates)

    def sample_joint(self, test_features: Tensor, num_samples: int=1) -> \
            Tensor:
        return sample_posterior_joint(
            *self._state_args(), test_features, num_samples)

    def sample_marginals(self, test_features: Tensor, num_samples: int=1) -> \
            Tensor:
        return sample_posterior_marginals(
            *self._state_args(), test_features, num_samples)

    def neg_log_likelihood(self) -> Tensor:
        """
        Works only if fantasy samples are not used (single targets vector).

        :return: Negative log (marginal) likelihood
        """
        return negative_log_marginal_likelihood(
            self.F, self.chol_fact, self.pred_mat)

    def is_ndarray(self):
        return mxnet_is_ndarray(self.F)

    def _check_is_ndarray(self):
        assert self.is_ndarray(), \
            "Implemented only for mxnet.ndarray, not for mxnet.symbol"


class IncrementalUpdateGPPosteriorState(GaussProcPosteriorState):
    """
    Extension of GaussProcPosteriorState which allows for incremental
    updating, given that a single data case is appended to the training
    set.

    In order to not mutate members, the update method returns a new
    object.

    """
    def __init__(
            self, features: Tensor, targets: Optional[Tensor],
            mean: MeanFunction, kernel: KernelFunction,
            noise_variance: Tensor, **kwargs):
        super(IncrementalUpdateGPPosteriorState, self).__init__(
            features, targets, mean, kernel, noise_variance, **kwargs)
        # Noise variance is needed here for updates (make copy, to be safe)
        self.noise_variance = noise_variance if self.F == mx.sym \
            else noise_variance.copy()

    def update(self, feature: Tensor, target: Tensor) -> \
            'IncrementalUpdateGPPosteriorState':
        """
        :param feature: Additional input xstar, shape (1, d)
        :param target: Additional target ystar, shape (1, m)
        :return: Posterior state for increased data set
        """
        # Ensure feature, target have one row
        F = self.F
        feature = F.reshape(feature, shape=(1, -1))
        target = F.reshape(target, shape=(1, -1))
        if self.is_ndarray():
            assert feature.shape[1] == self.features.shape[1], \
                "feature.shape[1] = {} != {} = self.features.shape[1]".format(
                    feature.shape[1], self.features.shape[1])
            assert target.shape[1] == self.pred_mat.shape[1], \
                "target.shape[1] = {} != {} = self.pred_mat.shape[1]".format(
                    target.shape[1], self.pred_mat.shape[1])
        chol_fact_new, pred_mat_new = cholesky_update(
            F, self.features, self.chol_fact, self.pred_mat, self.mean,
            self.kernel, self.noise_variance, feature, target)
        features_new = F.concat(self.features, feature, dim=0)
        # Create object by calling internal constructor
        state_new = IncrementalUpdateGPPosteriorState(
            features=features_new,
            targets=None,
            mean=self.mean,
            kernel=self.kernel,
            noise_variance=self.noise_variance,
            chol_fact=chol_fact_new,
            pred_mat=pred_mat_new)
        return state_new

    def expand_fantasies(self, num_fantasies: int) -> \
            'IncrementalUpdateGPPosteriorState':
        """
        If this posterior has been created with a single targets vector,
        shape (n, 1), use this to duplicate this vector m = num_fantasies
        times. Call this method before fantasy targets are appended by
        update.

        :param num_fantasies: Number m of fantasy samples
        :return: New state with targets duplicated m times
        """
        assert num_fantasies > 1
        F = self.F
        if self.is_ndarray():
            assert self.pred_mat.shape[1] == 1, \
                "Method requires posterior without fantasy samples"
        pred_mat_new = F.concat(*([self.pred_mat] * num_fantasies), dim=1)
        state_new = IncrementalUpdateGPPosteriorState(
            features=self.features,
            targets=None,
            mean=self.mean,
            kernel=self.kernel,
            noise_variance=self.noise_variance,
            chol_fact=self.chol_fact,
            pred_mat=pred_mat_new)
        return state_new
