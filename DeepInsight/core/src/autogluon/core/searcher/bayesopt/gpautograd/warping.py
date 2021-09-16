import autograd.numpy as anp

from .constants import DEFAULT_ENCODING, INITIAL_WARPING, WARPING_LOWER_BOUND, \
    WARPING_UPPER_BOUND, NUMERICAL_JITTER
from .distribution import LogNormal
from .kernel import KernelFunction
from .gluon import Block
from .gluon_blocks_helpers import encode_unwrap_parameter
from .utils import create_encoding, register_parameter


class OneDimensionalWarping(Block):
    """
    Block that is responsible for the warping of a single, column
    feature x. Typically, the full data X = [x1, x2,..., xd] is in (n, d) and
    each xi is a column feature in (n, 1).

    Consider column feature x and assume that the entries of x are contained in
    the range input_range. Each entry of x is transformed by
        warping(u) = 1. - (1. - R(u)^a)^b,
    with a,b two non negative parameters learned by empirical Bayes, and R(.)
    is a linear transformation that, based on input_range, rescales the entry
    of x into [eps, 1-eps] for some small eps > 0.

    :param input_range: tuple that contains the lower and upper bounds of the
        entries of x.
    """
    def __init__(self, input_range, encoding_type=DEFAULT_ENCODING, **kwargs):
        super().__init__(**kwargs)
        self.input_range = input_range
        self.encoding = create_encoding(
            encoding_type, INITIAL_WARPING, WARPING_LOWER_BOUND,
            WARPING_UPPER_BOUND, 2, LogNormal(0.0, 0.75))
        with self.name_scope():
            self.warping_internal = register_parameter(
                self.params, 'warping', self.encoding, shape=(2,))

    def _rescale(self, x):
        """
        We linearly rescale the entries of x into [NUMERICAL_JITTER, 1-NUMERICAL_JITTER]
        In this way, we avoid the differentiability problems at 0
        :param x: np.ndarray to be rescaled
        """
        lower, upper = self.input_range
        P = (1. - 2 * NUMERICAL_JITTER)/(upper - lower)
        Q = (NUMERICAL_JITTER * (upper + lower) - lower)/(upper - lower)

        return P * x + Q

    def _warping(self):
        return encode_unwrap_parameter(self.warping_internal, self.encoding)

    def forward(self, x):
        """
        Actual computation of the warping transformation (see details above)

        :param x: input data of size (n,1)
        """
        warping = anp.reshape(self._warping(), (-1,))
        warping_a = warping[0]
        warping_b = warping[1]

        return 1. - anp.power(1. - anp.power(
            self._rescale(x), warping_a), warping_b)

    def param_encoding_pairs(self):
        """
        Return a list of tuples with the Gluon parameters of the 1-D warping
        and their respective encodings
        """
        return [(self.warping_internal, self.encoding)]
    
    def get_params(self):
        warping = anp.reshape(self._warping(), (-1,))
        return {
            'warping_a': warping[0],
            'warping_b': warping[1]}

    def set_params(self, param_dict):
        warping = [param_dict['warping_a'], param_dict['warping_b']]
        self.encoding.set(self.warping_internal, warping)


class Warping(Block):
    """
    Block that computes warping over all the columns of some input data X.
    If X is of size (n,dimension), where dimension has to be specified, a 1-D warping
    transformation is applied to each column X[:,j] with j a key in index_to_range.
    More precisely, index_to_range is a dictionary of the form
        {
            j : (lower_bound_column_j, upper_bound_column_j),
            k : (lower_bound_column_k, upper_bound_column_k),
            ....
        }
    that maps column indexes to their corresponding ranges.
    """
    def __init__(self, dimension, index_to_range, encoding_type=DEFAULT_ENCODING,
                 **kwargs):
        super().__init__(**kwargs)

        assert isinstance(index_to_range, dict)
        assert all(isinstance(r, tuple) for r in index_to_range.values())
        assert all(r[0] < r[1] for r in index_to_range.values()) # for dictionary

        self.transformations = []
        self._params_encoding_pairs = []
        self.dimension = dimension
        self.index_to_range = index_to_range

        some_are_warped = False
        for col_index in range(dimension):
            if col_index in index_to_range:
                transformation = OneDimensionalWarping(
                    index_to_range[col_index], encoding_type=encoding_type)
                # To make sure that OneDimensionalWarping will get initialized
                # and managed by Warping, we register it as a child.
                self.register_child(transformation, name=transformation.name)
                self._params_encoding_pairs += transformation.param_encoding_pairs()
                some_are_warped = True
            else:
                # if a column is not warped, we do not apply any transformation
                transformation = lambda x: x
            self.transformations.append(transformation)
        assert some_are_warped,\
            "At least one of the dimensions must be warped"

    def forward(self, X):
        """
        Actual computation of warping applied to each column of X

        :param X: input data of size (n,dimension)
        """
        warped_X = []
        for col_index, transformation in enumerate(self.transformations):
            x = X[:, col_index:(col_index+1)]
            warped_X.append(transformation(x))

        return anp.concatenate(warped_X, axis=1)

    def param_encoding_pairs(self):
        """
        Return a list of tuples with the Gluon parameters of the warping and
        their respective encodings
        """
        return self._params_encoding_pairs

    def get_params(self):
        """
        Keys are warping_a, warping_b if there is one dimension, and
        warping_a<k>, warping_b<k> otherwise.
        """
        if len(self.transformations) == 1:
            result = self.transformations[0].get_params()
        else:
            result = dict()
            for i, warping in enumerate(self.transformations):
                if isinstance(warping, OneDimensionalWarping):
                    istr = str(i)
                    for k, v in warping.get_params().items():
                        result[k + istr] = v
        return result

    def set_params(self, param_dict):
        if len(self.transformations) == 1:
            self.transformations[0].set_params(param_dict)
        else:
            transf_keys = None
            for i, warping in enumerate(self.transformations):
                if isinstance(warping, OneDimensionalWarping):
                    if transf_keys is None:
                        transf_keys = warping.get_params().keys()
                    istr = str(i)
                    stripped_dict = dict()
                    for k in transf_keys:
                        stripped_dict[k] = param_dict[k + istr]
                    warping.set_params(stripped_dict)


class WarpedKernel(KernelFunction):
    """
    Block that composes warping with an arbitrary kernel
    """
    def __init__(self, kernel: KernelFunction, warping: Warping, **kwargs):
        super().__init__(kernel.dimension, **kwargs)
        self.kernel = kernel
        self.warping = warping

    def forward(self, X1, X2):
        """
        Actual computation of the composition of warping with an arbitrary
        kernel K. If we have input data X1 and X2, of respective dimensions
        (n1, d) and (n2, d), we compute the matrix

            K(warping(X1), warping(X2)) of size (n1,n2)
            whose (i,j) entry is given by K(warping(X1[i,:]), warping(X2[j,:]))

        :param X1: input data of size (n1, d)
        :param X2: input data of size (n2, d)
        """
        warped_X1 = self.warping(X1)
        if X1 is X2:
            warped_X2 = warped_X1
        else:
            warped_X2 = self.warping(X2)
        return self.kernel(warped_X1, warped_X2)

    def diagonal(self, X):
        # If kernel.diagonal does not depend on content of X (but just its
        # size), can pass X instead of self.warping(X)
        warped_X = self.warping(X) if self.kernel.diagonal_depends_on_X()             else X
        return self.kernel.diagonal(warped_X)

    def diagonal_depends_on_X(self):
        return self.kernel.diagonal_depends_on_X()

    def param_encoding_pairs(self):
        return self.kernel.param_encoding_pairs() + \
               self.warping.param_encoding_pairs()

    def get_params(self):
        # We use the union of get_params for kernel and warping, without
        # prefixes.
        result = self.kernel.get_params()
        result.update(self.warping.get_params())
        return result

    def set_params(self, param_dict):
        self.kernel.set_params(param_dict)
        self.warping.set_params(param_dict)
