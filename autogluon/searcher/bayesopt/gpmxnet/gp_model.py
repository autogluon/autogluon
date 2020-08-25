from abc import ABC, abstractmethod
from typing import List, Optional
import mxnet as mx

from .constants import DATA_TYPE
from .posterior_state import GaussProcPosteriorState


class GaussianProcessModel(ABC):
    @property
    @abstractmethod
    def states(self) -> Optional[List[GaussProcPosteriorState]]:
        pass

    @property
    @abstractmethod
    def ctx(self):
        pass

    @abstractmethod
    def fit(self, X: mx.nd.NDArray, Y: mx.nd.NDArray):
        """Train GP on the data and set a list of posterior states to be used by predict function"""
        pass

    @abstractmethod
    def recompute_states(self, X: mx.nd.NDArray, Y: mx.nd.NDArray):
        """Fixing GP hyperparameters and recompute the list of posterior states based on X and Y"""
        pass

    def _check_and_format_input(self, u):
        """
        Check and massage the input to conform with the numerical type and context

        :param u: some mx.nd
        """
        assert isinstance(u, mx.nd.NDArray)
        if u.ndim == 1:
            u = u.reshape(-1, 1)
        if u.dtype != DATA_TYPE or u.context != self.ctx:
            return mx.nd.array(u, dtype=DATA_TYPE, ctx=self.ctx)
        else:
            return u

    def predict(self, X_test):
        """
        Compute the posterior mean(s) and variance(s) for the points in X_test.
        If the posterior state is based on m target vectors, a (n, m) matrix is
        returned for posterior means.

        :param X_test: Data matrix X_test of size (n, d) (type mx.nd) for which n
            predictions are made
        :return: posterior_means, posterior_variances
        """

        X_test = self._assert_check_xtest(X_test)
        predictions = []
        for state in self.states:
            post_means, post_vars = state.predict(X_test)
            # Just to make sure the return shapes are the same as before:
            if post_means.shape[1] == 1:
                post_means = post_means.reshape((-1,))
            predictions.append((post_means, post_vars))
        return predictions

    def _assert_check_xtest(self, X_test):
        assert self.states is not None, \
            "Posterior state does not exist (run 'fit')"
        X_test = self._check_and_format_input(X_test)
        assert X_test.shape[1] == self.states[0].num_features, \
            "X_test and X_train should have the same number of columns (received {}, expected {})".format(
                X_test.shape[1], self.states[0].num_features)
        return X_test

    def multiple_targets(self):
        """
        :return: Posterior state based on multiple (fantasized) target vectors?
        """
        assert self.states is not None, \
            "Posterior state does not exist (run 'fit')"
        return self.states[0].num_fantasies > 1

    def sample_marginals(self, X_test, num_samples=1):
        """
        Draws marginal samples from predictive distribution at n test points.
        Notice we concat the samples for each state. Let n_states = len(self._states)

        If the posterior state is based on m > 1 target vectors, a
        (n, m, num_samples * n_states) tensor is returned, for m == 1 we return a
        (n, num_samples * n_states) matrix.

        :param X_test: Test input points, shape (n, d)
        :param num_samples: Number of samples
        :return: Samples with shape (n, num_samples * n_states) or (n, m, num_samples * n_states) if m > 1
        """

        X_test = self._assert_check_xtest(X_test)
        samples_list = [state.sample_marginals(X_test, num_samples)
                        for state in self.states]
        return _concatenate_samples(samples_list)

    def sample_joint(self, X_test, num_samples=1):
        """
        Draws joint samples from predictive distribution at n test points.
        This scales cubically with n.
        the posterior state must be based on a single target vector
        (m > 1 is not supported).

        :param X_test: Test input points, shape (n, d)
        :param num_samples: Number of samples
        :return: Samples, shape (n, num_samples)
        """

        X_test = self._assert_check_xtest(X_test)
        samples_list = [state.sample_joint(X_test, num_samples)
                        for state in self.states]
        return _concatenate_samples(samples_list)


def _concatenate_samples(samples_list: List[mx.nd.NDArray]) -> mx.nd.NDArray:
    return mx.nd.concat(*samples_list, dim=-1)
