import numpy as np
import mxnet as mx

from autogluon.searcher.bayesopt.gpmxnet.posterior_state import \
    IncrementalUpdateGPPosteriorState, GaussProcPosteriorState
from autogluon.searcher.bayesopt.gpmxnet.gp_regression import \
    GaussianProcessRegression
from autogluon.searcher.bayesopt.gpmxnet.kernel import Matern52


def to_nd(x, dtype=np.float64):
    return mx.nd.array(x, dtype=dtype)


def test_incremental_update():
    def f(x):
        return np.sin(x) / x

    np.random.seed(298424)
    std_noise = 0.01

    for rep in range(10):
        model = GaussianProcessRegression(kernel=Matern52(dimension=1))
        # Sample data
        num_train = np.random.randint(low=5, high=15)
        num_incr = np.random.randint(low=1, high=7)
        sizes = [num_train, num_incr]
        features = []
        targets = []
        for sz in sizes:
            feats = np.random.uniform(low=-1.0, high=1.0, size=sz).reshape((-1, 1))
            features.append(feats)
            targs = f(feats)
            targs += np.random.normal(0.0, std_noise, size=targs.shape)
            targets.append(targs)
        # Posterior state by incremental updating
        train_features = to_nd(features[0])
        train_targets = to_nd(targets[0])
        model.fit(train_features, train_targets)
        noise_variance_1 = model.likelihood.get_noise_variance()
        state_incr = IncrementalUpdateGPPosteriorState(
            features=train_features, targets=train_targets,
            mean=model.likelihood.mean, kernel=model.likelihood.kernel,
            noise_variance=model.likelihood.get_noise_variance(as_ndarray=True))
        for i in range(num_incr):
            state_incr = state_incr.update(
                to_nd(features[1][i].reshape((1, -1))),
                to_nd(targets[1][i].reshape((1, -1))))
        noise_variance_2 = state_incr.noise_variance.asscalar()
        # Posterior state by direct computation
        state_comp = GaussProcPosteriorState(
            features=to_nd(np.concatenate(features, axis=0)),
            targets=to_nd(np.concatenate(targets, axis=0)),
            mean=model.likelihood.mean, kernel=model.likelihood.kernel,
            noise_variance=state_incr.noise_variance)
        # Compare them
        assert noise_variance_1 == noise_variance_2, \
            "noise_variance_1 = {} != {} = noise_variance_2".format(
                noise_variance_1, noise_variance_2)
        chol_fact_incr = state_incr.chol_fact.asnumpy()
        chol_fact_comp = state_comp.chol_fact.asnumpy()
        np.testing.assert_almost_equal(chol_fact_incr, chol_fact_comp, decimal=2)
        pred_mat_incr = state_incr.pred_mat.asnumpy()
        pred_mat_comp = state_comp.pred_mat.asnumpy()
        np.testing.assert_almost_equal(pred_mat_incr, pred_mat_comp, decimal=2)


if __name__ == "__main__":
    test_incremental_update()
