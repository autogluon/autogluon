import mxnet as mx
import logging

logger = logging.getLogger(__name__)

__all__ = ['AddJitterOp', 'AddJitterOpProp']


INITIAL_JITTER_FACTOR = 1e-9
JITTER_GROWTH = 10.
JITTER_UPPERBOUND_FACTOR = 1e3


class AddJitterOp(mx.operator.CustomOp):
    """
    Finds smaller jitter to add to diagonal of square matrix to render the
    matrix positive definite (in that linalg.potrf works).

    Given input x (positive semi-definite matrix) and sigsq_init (nonneg
    scalar), find sigsq_final (nonneg scalar), so that:
        sigsq_final = sigsq_init + jitter, jitter >= 0,
        x + sigsq_final * Id positive definite (so that potrf call works)
    We return the matrix x + sigsq_final * Id, for which potrf has not failed.

    For the gradient, the dependence of jitter on the inputs is ignored.

    The values tried for sigsq_final are:
        sigsq_init, sigsq_init + initial_jitter * (jitter_growth ** k),
        k = 0, 1, 2, ...,
        initial_jitter = initial_jitter_factor * mean(diag(x))

    Note: The scaling of initial_jitter with mean(diag(x)) is taken from GPy.
    The rationale is that the largest eigenvalue of x is >= mean(diag(x)), and
    likely of this magnitude.

    There is no guarantee that the Cholesky factor returned is well-conditioned
    enough for subsequent computations to be reliable. A better solution
    would be to estimate the condition number of the Cholesky factor, and to add
    jitter until this is bounded below a threshold we tolerate. See

        Higham, N.
        A Survey of Condition Number Estimation for Triangular Matrices
        MIMS EPrint: 2007.10

    Algorithm 4.1 could work for us.
    """

    def __init__(
            self, initial_jitter_factor, jitter_growth, debug_log, **kwargs):
        super(AddJitterOp, self).__init__(**kwargs)

        assert initial_jitter_factor > 0. and jitter_growth > 1.
        self._initial_jitter_factor = initial_jitter_factor
        self._jitter_growth = jitter_growth
        self._debug_log = debug_log

    def _get_constant_identity(self, x, constant):
        n, _ = x.shape
        return mx.nd.diag(
            mx.nd.ones(shape=(n,), ctx=x.context, dtype=x.dtype) * constant)

    def _get_jitter_upperbound(self, x):
        # To define a safeguard in the while-loop of the forward,
        # we define an upperbound on the jitter we can reasonably add
        # the bound is quite generous, and is dependent on the scale of the input x
        # (the scale is captured via the trace of x)
        # the primary goal is avoid any infinite while-loop.
        return JITTER_UPPERBOUND_FACTOR * max(
            1., mx.nd.mean(mx.nd.diag(x)).asscalar())

    def forward(self, is_train, req, in_data, out_data, aux):
        x = in_data[0]
        sigsq_init = in_data[1]
        jitter = 0.
        jitter_upperbound = self._get_jitter_upperbound(x)
        must_increase_jitter = True
        x_plus_constant = None
        while must_increase_jitter and jitter <= jitter_upperbound:
            try:
                x_plus_constant = x + self._get_constant_identity(
                    x, sigsq_init + jitter)
                L = mx.nd.linalg.potrf(x_plus_constant)
                # because of the implicit asynchronous processing in MXNet,
                # we need to enforce the computation of L to happen right here
                L.wait_to_read()
                must_increase_jitter = False
            except mx.base.MXNetError:
                if self._debug_log == 'true':
                    logger.info("sigsq = {} does not work".format(
                        sigsq_init.asscalar() + jitter))
                if jitter == 0.0:
                    jitter = self._initial_jitter_factor * mx.nd.mean(
                        mx.nd.diag(x)).asscalar()
                else:
                    jitter *= self._jitter_growth

        assert not must_increase_jitter,\
            "The jitter ({}) has reached its upperbound ({}) while the Cholesky of the input matrix still cannot be computed.".format(jitter, jitter_upperbound)
        if self._debug_log == 'true':
            _sigsq_init = sigsq_init.asscalar()
            logger.info("sigsq_final = {}".format(_sigsq_init + jitter))
        self.assign(out_data[0], req[0], x_plus_constant)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        self.assign(in_grad[0], req[0], out_grad[0])
        trace_out_grad = mx.nd.sum(mx.nd.diag(out_grad[0]))
        self.assign(in_grad[1], req[1], trace_out_grad)


@mx.operator.register("add_jitter")
class AddJitterOpProp(mx.operator.CustomOpProp):
    def __init__(
            self, initial_jitter_factor=INITIAL_JITTER_FACTOR,
            jitter_growth=JITTER_GROWTH, debug_log='false'):
        super(AddJitterOpProp, self).__init__(need_top_grad=True)
        # We need to cast the arguments
        # see detailed example https://github.com/Xilinx/mxnet/blob/master/docs/tutorials/gluon/customop.md
        self._initial_jitter_factor = float(initial_jitter_factor)
        self._jitter_growth = float(jitter_growth)
        self._debug_log = debug_log

    def list_arguments(self):
        return ['x', 'sigsq_init']

    def list_outputs(self):
        return ['x_plus_sigsq_final']

    def infer_shape(self, in_shape):
        x_shape = in_shape[0]
        assert len(x_shape) == 2 and x_shape[0] == x_shape[1], \
            "x must be square matrix, shape (n, n)"
        ssq_shape = in_shape[1]
        assert len(ssq_shape) == 1 and ssq_shape[0] == 1, \
            "sigsq_init must be scalar, shape (1,)"
        return in_shape, [x_shape], []

    def create_operator(self, ctx, shapes, dtypes, **kwargs):
        return AddJitterOp(
            initial_jitter_factor=self._initial_jitter_factor,
            jitter_growth=self._jitter_growth,
            debug_log=self._debug_log, **kwargs)
