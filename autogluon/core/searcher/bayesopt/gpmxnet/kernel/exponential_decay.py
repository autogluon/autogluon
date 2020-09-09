import mxnet as mx

from .base import KernelFunction
from ..constants import DEFAULT_ENCODING
from ..gluon_blocks_helpers import unwrap_parameter, IdentityScalarEncoding
from ..mean import MeanFunction
from ..utils import create_encoding, register_parameter, get_name_internal

__all__ = ['ExponentialDecayResourcesKernelFunction',
           'ExponentialDecayResourcesMeanFunction']


class ExponentialDecayResourcesKernelFunction(KernelFunction):
    """
    Variant of the kernel function for modeling exponentially decaying
    learning curves, proposed in:

        Swersky, K., Snoek, J., & Adams, R. P. (2014).
        Freeze-Thaw Bayesian Optimization.
        ArXiv:1406.3896 [Cs, Stat).
        Retrieved from http://arxiv.org/abs/1406.3896

    The argument in that paper actually justifies using a non-zero mean
    function (see ExponentialDecayResourcesMeanFunction) and centralizing
    the kernel proposed there. This is done here. Details in:

        Tiao, Klein, Archambeau, Seeger (2020)
        Model-based Asynchronous Hyperparameter Optimization
        https://arxiv.org/abs/2003.10865

    We implement a new family of kernel functions, for which the additive
    Freeze-Thaw kernel is one instance (delta = 0).
    The kernel has parameters alpha, mean_lam, gamma > 0, and delta in [0, 1].
    Note that beta = alpha / mean_lam is used in the Freeze-Thaw paper (the
    Gamma distribution over lambda is parameterized differently).
    The additive Freeze-Thaw kernel is obtained for delta = 0 (use
    delta_fixed_value = 0).

    In fact, this class is configured with a kernel and a mean function over
    inputs x (dimension d) and represents a kernel (and mean function) over
    inputs (x, r) (dimension d + 1), where the resource attribute r >= 0 is
    last.

    """
    def __init__(
            self, kernel_x: KernelFunction, mean_x: MeanFunction,
            encoding_type=DEFAULT_ENCODING, alpha_init=1.0, mean_lam_init=0.5,
            gamma_init=0.5, delta_fixed_value=None, delta_init=0.5,
            max_metric_value=1.0, **kwargs):
        """
        :param kernel_x: Kernel k_x(x, x') over configs
        :param mean_x: Mean function mu_x(x) over configs
        :param encoding_type: Encoding used for alpha, mean_lam, gamma (positive
            values)
        :param alpha_init: Initial value alpha
        :param mean_lam_init: Initial value mean_lam
        :param gamma_init: Initial value gamma
        :param delta_fixed_value: If not None, delta is fixed to this value, and
            does not become a free parameter
        :param delta_init: Initial value delta (if delta_fixed_value is None)
        :param max_metric_value: Maximum value which metric can attend. This is
            used as upper bound on gamma
        """

        super(ExponentialDecayResourcesKernelFunction, self).__init__(
            dimension=kernel_x.dimension + 1, **kwargs)
        self.kernel_x = kernel_x
        self.mean_x = mean_x
        # alpha, mean_lam are parameters of a Gamma distribution, where alpha is
        # a scale parameter, and
        #   E[lambda] = mean_lam, Var[lambda] = mean_lam ** 2 / alpha
        alpha_lower, alpha_upper = 1e-6, 250.0
        alpha_init = self._wrap_initvals(alpha_init, alpha_lower, alpha_upper)
        self.encoding_alpha = create_encoding(
            encoding_type, alpha_init, alpha_lower, alpha_upper, 1, None)
        mean_lam_lower, mean_lam_upper = 1e-4, 50.0
        mean_lam_init = self._wrap_initvals(
            mean_lam_init, mean_lam_lower, mean_lam_upper)
        self.encoding_mean_lam = create_encoding(
            encoding_type, mean_lam_init, mean_lam_lower, mean_lam_upper, 1,
            None)
        # If f(x, 0) is the metric value at r -> 0, f(x) at r -> infty,
        # then f(x, 0) = gamma (for delta = 1), or f(x, 0) = gamma + f(x) for
        # delta = 0. gamma should not be largest than the maximum metric
        # value.
        gamma_lower= max_metric_value * 0.0001
        gamma_upper = max_metric_value
        gamma_init = self._wrap_initvals(gamma_init, gamma_lower, gamma_upper)
        self.encoding_gamma = create_encoding(
            encoding_type, gamma_init, gamma_lower, gamma_upper, 1, None)
        if delta_fixed_value is None:
            delta_init = self._wrap_initvals(delta_init, 0.0, 1.0)
            self.encoding_delta = IdentityScalarEncoding(
                constr_lower=0.0, constr_upper=1.0, init_val=delta_init)
        else:
            assert 0.0 <= delta_fixed_value <= 1.0, \
                "delta_fixed_value = {}, must lie in [0, 1]".format(
                    delta_fixed_value)
            self.encoding_delta = None
            self.delta_fixed_value = delta_fixed_value

        with self.name_scope():
            self.alpha_internal = register_parameter(
                self.params, "alpha", self.encoding_alpha)
            self.mean_lam_internal = register_parameter(
                self.params, "mean_lam", self.encoding_mean_lam)
            self.gamma_internal = register_parameter(
                self.params, "gamma", self.encoding_gamma)
            if delta_fixed_value is None:
                self.delta_internal = register_parameter(
                    self.params, "delta", self.encoding_delta)

    @staticmethod
    def _wrap_initvals(init, lower, upper):
        return max(min(init, upper * 0.999), lower * 1.001)

    @staticmethod
    def _compute_kappa(F, x, alpha, mean_lam):
        beta = alpha / mean_lam
        return F.broadcast_power(F.broadcast_div(
            beta, F.broadcast_add(x, beta)), alpha)

    def _compute_terms(self, F, X, alpha, mean_lam, gamma, delta, ret_mean=False):
        dim = self.kernel_x.dimension
        cfg = F.slice_axis(X, axis=1, begin=0, end=dim)
        res = F.slice_axis(X, axis=1, begin=dim, end=None)
        kappa = self._compute_kappa(F, res, alpha, mean_lam)
        kr_pref = F.reshape(gamma, shape=(1, 1))
        if ret_mean or (self.encoding_delta is not None) or delta > 0.0:
            mean = self.mean_x(cfg)
        else:
            mean = None
        if self.encoding_delta is not None:
            kr_pref = F.broadcast_sub(kr_pref, F.broadcast_mul(delta, mean))
        elif delta > 0.0:
            kr_pref = F.broadcast_sub(kr_pref, mean * delta)
        return cfg, res, kappa, kr_pref, mean

    @staticmethod
    def _unwrap(F, X, kwargs, key, enc, var_internal):
        return enc.get(
            F, kwargs.get(get_name_internal(key),
                          unwrap_parameter(F, var_internal, X)))

    def _get_params(self, F, X, **kwargs):
        alpha = self._unwrap(
            F, X, kwargs, 'alpha', self.encoding_alpha, self.alpha_internal)
        mean_lam = self._unwrap(
            F, X, kwargs, 'mean_lam', self.encoding_mean_lam,
            self.mean_lam_internal)
        gamma = self._unwrap(
            F, X, kwargs, 'gamma', self.encoding_gamma, self.gamma_internal)
        if self.encoding_delta is not None:
            delta = F.reshape(self._unwrap(
                F, X, kwargs, 'delta', self.encoding_delta,
                self.delta_internal), shape=(1, 1))
        else:
            delta = self.delta_fixed_value
        return (alpha, mean_lam, gamma, delta)

    def hybrid_forward(self, F, X1, X2, **kwargs):
        alpha, mean_lam, gamma, delta = self._get_params(F, X1, **kwargs)

        cfg1, res1, kappa1, kr_pref1, _ = self._compute_terms(
            F, X1, alpha, mean_lam, gamma, delta)
        if X2 is not X1:
            cfg2, res2, kappa2, kr_pref2, _ = self._compute_terms(
                F, X2, alpha, mean_lam, gamma, delta)
        else:
            cfg2, res2, kappa2, kr_pref2 = cfg1, res1, kappa1, kr_pref1
        res2 = F.reshape(res2, shape=(1, -1))
        kappa2 = F.reshape(kappa2, shape=(1, -1))
        kr_pref2 = F.reshape(kr_pref2, shape=(1, -1))
        kappa12 = self._compute_kappa(
            F, F.broadcast_add(res1, res2), alpha, mean_lam)

        kmat_res = F.broadcast_sub(kappa12, F.broadcast_mul(kappa1, kappa2))
        kmat_res = F.broadcast_mul(kr_pref1, F.broadcast_mul(
            kr_pref2, kmat_res))
        kmat_x = self.kernel_x(cfg1, cfg2)
        if self.encoding_delta is None:
            if delta > 0.0:
                tmpmat = F.broadcast_add(kappa1, F.broadcast_sub(
                    kappa2, kappa12 * delta))
                tmpmat = tmpmat * (-delta) + 1.0
            else:
                tmpmat = 1.0
        else:
            tmpmat = F.broadcast_add(kappa1, F.broadcast_sub(
                kappa2, F.broadcast_mul(kappa12, delta)))
            tmpmat = F.broadcast_mul(tmpmat, -delta) + 1.0

        return kmat_x * tmpmat + kmat_res

    def diagonal(self, F, X):
        alpha, mean_lam, gamma, delta = self._get_params(F, X)

        cfg, res, kappa, kr_pref, _ = self._compute_terms(
            F, X, alpha, mean_lam, gamma, delta)
        kappa2 = self._compute_kappa(F, res * 2, alpha, mean_lam)

        kdiag_res = F.broadcast_sub(kappa2, F.square(kappa))
        kdiag_res = F.reshape(
            F.broadcast_mul(kdiag_res, F.square(kr_pref)), shape=(-1,))
        kdiag_x = self.kernel_x.diagonal(F, cfg)
        if self.encoding_delta is None:
            if delta > 0.0:
                tmpvec = F.broadcast_sub(kappa * 2, kappa2 * delta)
                tmpvec = F.reshape(tmpvec * (-delta) + 1.0, shape=(-1,))
            else:
                tmpvec = 1.0
        else:
            tmpvec = F.broadcast_sub(kappa * 2, F.broadcast_mul(kappa2, delta))
            tmpvec = F.reshape(
                F.broadcast_mul(tmpvec, -delta) + 1.0, shape=(-1,))

        return kdiag_x * tmpvec + kdiag_res

    def diagonal_depends_on_X(self):
        return True

    def param_encoding_pairs(self):
        enc_list = [
            (self.alpha_internal, self.encoding_alpha),
            (self.mean_lam_internal, self.encoding_mean_lam),
            (self.gamma_internal, self.encoding_gamma)]
        if self.encoding_delta is not None:
            enc_list.append((self.delta_internal, self.encoding_delta))
        enc_list.extend(self.kernel_x.param_encoding_pairs())
        enc_list.extend(self.mean_x.param_encoding_pairs())
        return enc_list

    def mean_function(self, F, X):
        alpha, mean_lam, gamma, delta = self._get_params(F, X)
        cfg, res, kappa, kr_pref, mean = self._compute_terms(
            F, X, alpha, mean_lam, gamma, delta, ret_mean=True)

        return F.broadcast_add(mean, F.broadcast_mul(kappa, kr_pref))

    def get_params(self):
        """
        Parameter keys are alpha, mean_lam, gamma, delta (only if not fixed
        to delta_fixed_value), as well as those of self.kernel_x (prefix
        'kernelx_') and of self.mean_x (prefix 'meanx_').

        """
        values = list(self._get_params(mx.nd, None))
        keys = ['alpha', 'mean_lam', 'gamma', 'delta']
        if self.encoding_delta is None:
            values.pop()
            keys.pop()
        result = {k: v.reshape((1,)).asscalar() for k, v in zip(keys, values)}
        for pref, func in [('kernelx_', self.kernel_x), ('meanx_', self.mean_x)]:
            result.update({
                (pref + k): v for k, v in func.get_params().items()})
        return result

    def set_params(self, param_dict):
        for pref, func in [('kernelx_', self.kernel_x), ('meanx_', self.mean_x)]:
            len_pref = len(pref)
            stripped_dict = {
                k[len_pref:]: v for k, v in param_dict.items()
                if k.startswith(pref)}
            func.set_params(stripped_dict)
        self.encoding_alpha.set(self.alpha_internal, param_dict['alpha'])
        self.encoding_mean_lam.set(
            self.mean_lam_internal, param_dict['mean_lam'])
        self.encoding_gamma.set(self.gamma_internal, param_dict['gamma'])
        if self.encoding_delta is not None:
            self.encoding_delta.set(self.delta_internal, param_dict['delta'])


class ExponentialDecayResourcesMeanFunction(MeanFunction):
    def __init__(self, kernel: ExponentialDecayResourcesKernelFunction,
                 **kwargs):
        super(ExponentialDecayResourcesMeanFunction, self).__init__(**kwargs)
        assert isinstance(kernel, ExponentialDecayResourcesKernelFunction)
        self.kernel = kernel

    def hybrid_forward(self, F, X):
        return self.kernel.mean_function(F, X)

    def param_encoding_pairs(self):
        return []

    def get_params(self):
        return dict()

    def set_params(self, param_dict):
        pass
