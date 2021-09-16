import autograd.numpy as anp

from .base import KernelFunction

__all__ = ['ProductKernelFunction']


class ProductKernelFunction(KernelFunction):
    """
    Given two kernel functions K1, K2, this class represents the product kernel
    function given by

        ((x1, x2), (y1, y2)) -> K(x1, y1) * K(x2, y2)

    We assume that parameters of K1 and K2 are disjoint.
    """
    def __init__(self, kernel1: KernelFunction, kernel2: KernelFunction,
                 name_prefixes=None, **kwargs):
        """
        :param kernel1: Kernel function K1
        :param kernel2: Kernel function K2
        :param name_prefixes: Name prefixes for K1, K2 used in get_params
        """
        super(ProductKernelFunction, self).__init__(
            kernel1.dimension + kernel2.dimension, **kwargs)
        self.kernel1 = kernel1
        self.kernel2 = kernel2
        if name_prefixes is None:
            self.name_prefixes = ['kernel1', 'kernel2']
        else:
            assert len(name_prefixes) == 2
            self.name_prefixes = name_prefixes

    def forward(self, X1, X2):
        d1 = self.kernel1.dimension
        X1_1 = X1[:, :d1]
        X1_2 = X1[:, d1:]
        X2_1 = X2[:, :d1]
        X2_2 = X1[:, d1:]
        kmat1 = self.kernel1(X1_1, X2_1)
        kmat2 = self.kernel2(X1_2, X2_2)
        return kmat1 * kmat2

    def diagonal(self, X):
        d1 = self.kernel1.dimension
        X1 = X[:, :d1]
        X2 = X[:, d1:]
        diag1 = self.kernel1.diagonal(X1)
        diag2 = self.kernel2.diagonal(X2)
        return diag1 * diag2

    def diagonal_depends_on_X(self):
        return (self.kernel1.diagonal_depends_on_X() or
                self.kernel2.diagonal_depends_on_X())

    def param_encoding_pairs(self):
        """
        Note: We assume that K1 and K2 have disjoint parameters, otherwise
        there will be a redundancy here.
        """
        return self.kernel1.param_encoding_pairs() + \
               self.kernel2.param_encoding_pairs()

    def get_params(self):
        result = dict()
        prefs = [k + '_' for k in self.name_prefixes]
        for pref, kernel in zip(prefs, [self.kernel1, self.kernel2]):
            result.update({
                (pref + k): v for k, v in kernel.get_params().items()})
                
        return result

    def set_params(self, param_dict):
        prefs = [k + '_' for k in self.name_prefixes]
        for pref, kernel in zip(prefs, [self.kernel1, self.kernel2]):
            len_pref = len(pref)
            stripped_dict = {
                k[len_pref:]: v for k, v in param_dict.items()
                if k.startswith(pref)}
            kernel.set_params(stripped_dict)
