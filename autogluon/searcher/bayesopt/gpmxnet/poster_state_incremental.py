import mxnet as mx
from mxnet.ndarray import NDArray

from autogluon.searcher.bayesopt.gpmxnet.posterior_utils import \
    sample_and_cholesky_update
from autogluon.searcher.bayesopt.gpmxnet.posterior_state import \
    GaussProcPosteriorState
from autogluon.searcher.bayesopt.gpmxnet.kernel import KernelFunction
from autogluon.searcher.bayesopt.gpmxnet.mean import MeanFunction


class GPPosteriorStateIncrementalUpdater(object):
    """
    This class supports incremental updating of Gaussian process posterior
    states, as in IncrementalUpdateGPPosteriorState.

    Compared to IncrementalUpdateGPPosteriorState, repeated computations of
    updates in sequence of ever-larger datasets is sped up here:

    - Drawing a fantasy sample and updating the state is done together, which
      saves compute
    - Whenever a certain size n (inputs) is encountered for the first time,
      an executor is bound and maintained here. When called the next time for
      this size, the executor is reused. This speeds up repeated execution,
      since for example all memory is already allocated

    NOTE: The same updater must not be used by several simulation threads in
    parallel. This is because the input and output arguments of the executors
    cached here are used as members of the posterior states. This works fine,
    as long as an updater is used sequentially along a single thread, but
    leads to errors when threads run in parallel.

    For a weak check, the updater maintains the size n of the last recently
    used executor in last_recent_exec_n. The current n in sample_and_update
    must be > last_recent_exec_n.
    This means that once a thread ends, reset has to be called for the updater,
    which resets last_recent_exec_n, so the next recent thread can use it
    afterwards.

    """
    def __init__(self, mean: MeanFunction, kernel: KernelFunction):
        self.mean = mean
        self.kernel = kernel
        self._executors = dict()
        self.last_recent_exec_n = None

    def sample_and_update(
            self, features: NDArray, chol_fact: NDArray, pred_mat: NDArray,
            noise_variance: NDArray,
            feature: NDArray) -> (NDArray, GaussProcPosteriorState):
        """
        Samples target for feature from predictive distribution (without
        noise_variance), then updates the posterior state accordingly.

        :param features: Part of current poster_state
        :param chol_fact: Part of current poster_state
        :param pred_mat: Part of current poster_state
        :param noise_variance: Part of current poster_state
        :param feature: New feature, shape (1, d)
        :return: (target, poster_state_new)

        """
        # Get executor for this size (bind if not exists)
        args = {
            'features': features,
            'chol_fact': chol_fact,
            'pred_mat': pred_mat,
            'noise_variance': noise_variance,
            'feature': feature}
        executor = self._get_executor(args)
        chol_fact_new = executor.outputs[0]
        pred_mat_new = executor.outputs[1]
        features_new = executor.outputs[2]
        target = executor.outputs[3]
        # Run computation
        for k, v in args.items():
            executor.arg_dict[k][:] = v
        executor.forward(is_train=False)
        poster_state_new = GaussProcPosteriorState(
            features = features_new,
            targets=None,
            mean=self.mean,
            kernel=self.kernel,
            noise_variance=noise_variance,
            chol_fact=chol_fact_new,
            pred_mat=pred_mat_new)
        return target, poster_state_new

    def reset(self):
        """
        Call this method at the end of a simulation thread.
        """
        self.last_recent_exec_n = None

    def _get_executor(self, args):
        n = args['features'].shape[0]
        # Sanity check
        assert self.last_recent_exec_n is None or n > self.last_recent_exec_n, \
            "Updater is not used in sequence in a single simulation thread [n = {}, last_recent_exec_n = {}]".format(
                n, self.last_recent_exec_n)
        if n not in self._executors:
            # Bind executor to sample_and_cholesky_update
            ex_args = {
                k: mx.nd.zeros_like(v) for k, v in args.items()}
            executor = self._proc_s().bind(
                ctx=args['features'].context, grad_req='null', args=ex_args)
            self._executors[n] = executor
        else:
            executor = self._executors[n]
        self.last_recent_exec_n = n
        return executor

    def _proc_s(self):
        return mx.sym.Group(list(sample_and_cholesky_update(
            mx.sym, mx.sym.Variable('features'),
            mx.sym.Variable('chol_fact'), mx.sym.Variable('pred_mat'),
            self.mean, self.kernel, mx.sym.Variable('noise_variance'),
            mx.sym.Variable('feature'))))

