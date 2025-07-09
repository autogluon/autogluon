# taken from https://github.com/yandex-research/rtdl-num-embeddings/blob/main/package/rtdl_num_embeddings.py
"""On Embeddings for Numerical Features in Tabular Deep Learning."""

__version__ = '0.0.12'

__all__ = [
    'LinearEmbeddings',
    'LinearReLUEmbeddings',
    'PeriodicEmbeddings',
    'PiecewiseLinearEmbeddings',
    'PiecewiseLinearEncoding',
    'compute_bins',
]

import math
import warnings
from typing import Any, Literal, Optional, Union

try:
    import sklearn.tree as sklearn_tree
except ImportError:
    sklearn_tree = None

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.parameter import Parameter

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


def _check_input_shape(x: Tensor, expected_n_features: int) -> None:
    if x.ndim < 1:
        raise ValueError(
            f'The input must have at least one dimension, however: {x.ndim=}'
        )
    if x.shape[-1] != expected_n_features:
        raise ValueError(
            'The last dimension of the input was expected to be'
            f' {expected_n_features}, however, {x.shape[-1]=}'
        )


class LinearEmbeddings(nn.Module):
    """Linear embeddings for continuous features.

    **Shape**

    - Input: `(*, n_features)`
    - Output: `(*, n_features, d_embedding)`

    **Examples**

    >>> batch_size = 2
    >>> n_cont_features = 3
    >>> x = torch.randn(batch_size, n_cont_features)
    >>> d_embedding = 4
    >>> m = LinearEmbeddings(n_cont_features, d_embedding)
    >>> m.get_output_shape()
    torch.Size([3, 4])
    >>> m(x).shape
    torch.Size([2, 3, 4])
    """

    def __init__(self, n_features: int, d_embedding: int) -> None:
        """
        Args:
            n_features: the number of continuous features.
            d_embedding: the embedding size.
        """
        if n_features <= 0:
            raise ValueError(f'n_features must be positive, however: {n_features=}')
        if d_embedding <= 0:
            raise ValueError(f'd_embedding must be positive, however: {d_embedding=}')

        super().__init__()
        self.weight = Parameter(torch.empty(n_features, d_embedding))
        self.bias = Parameter(torch.empty(n_features, d_embedding))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        d_rqsrt = self.weight.shape[1] ** -0.5
        nn.init.uniform_(self.weight, -d_rqsrt, d_rqsrt)
        nn.init.uniform_(self.bias, -d_rqsrt, d_rqsrt)

    def get_output_shape(self) -> torch.Size:
        """Get the output shape without the batch dimensions."""
        return self.weight.shape

    def forward(self, x: Tensor) -> Tensor:
        """Do the forward pass."""
        _check_input_shape(x, self.weight.shape[0])
        return torch.addcmul(self.bias, self.weight, x[..., None])


class LinearReLUEmbeddings(nn.Module):
    """Simple non-linear embeddings for continuous features.

    **Shape**

    - Input: `(*, n_features)`
    - Output: `(*, n_features, d_embedding)`

    **Examples**

    >>> batch_size = 2
    >>> n_cont_features = 3
    >>> x = torch.randn(batch_size, n_cont_features)
    >>>
    >>> d_embedding = 32
    >>> m = LinearReLUEmbeddings(n_cont_features, d_embedding)
    >>> m.get_output_shape()
    torch.Size([3, 32])
    >>> m(x).shape
    torch.Size([2, 3, 32])
    """

    def __init__(self, n_features: int, d_embedding: int = 32) -> None:
        """
        Args:
            n_features: the number of continuous features.
            d_embedding: the embedding size.
        """
        super().__init__()
        self.linear = LinearEmbeddings(n_features, d_embedding)
        self.activation = nn.ReLU()

    def get_output_shape(self) -> torch.Size:
        """Get the output shape without the batch dimensions."""
        return self.linear.weight.shape

    def forward(self, x: Tensor) -> Tensor:
        """Do the forward pass."""
        x = self.linear(x)
        x = self.activation(x)
        return x


class _Periodic(nn.Module):
    """
    NOTE: THIS MODULE SHOULD NOT BE USED DIRECTLY.

    Technically, this is a linear embedding without bias followed by
    the periodic activations. The scale of the initialization
    (defined by the `sigma` argument) plays an important role.
    """

    def __init__(self, n_features: int, k: int, sigma: float) -> None:
        if sigma <= 0.0:
            raise ValueError(f'sigma must be positive, however: {sigma=}')

        super().__init__()
        self._sigma = sigma
        self.weight = Parameter(torch.empty(n_features, k))
        self.reset_parameters()

    def reset_parameters(self):
        """Reset the parameters."""
        # NOTE[DIFF]
        # Here, extreme values (~0.3% probability) are explicitly avoided just in case.
        # In the paper, there was no protection from extreme values.
        bound = self._sigma * 3
        nn.init.trunc_normal_(self.weight, 0.0, self._sigma, a=-bound, b=bound)

    def forward(self, x: Tensor) -> Tensor:
        """Do the forward pass."""
        _check_input_shape(x, self.weight.shape[0])
        x = 2 * math.pi * self.weight * x[..., None]
        x = torch.cat([torch.cos(x), torch.sin(x)], -1)
        return x


# _NLinear is a simplified copy of delu.nn.NLinear:
# https://yura52.github.io/delu/stable/api/generated/delu.nn.NLinear.html
class _NLinear(nn.Module):
    """N *separate* linear layers for N feature embeddings.

    In other words,
    each feature embedding is transformed by its own dedicated linear layer.
    """

    def __init__(
        self, n: int, in_features: int, out_features: int, bias: bool = True
    ) -> None:
        super().__init__()
        self.weight = Parameter(torch.empty(n, in_features, out_features))
        self.bias = Parameter(torch.empty(n, out_features)) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        """Reset the parameters."""
        d_in_rsqrt = self.weight.shape[-2] ** -0.5
        nn.init.uniform_(self.weight, -d_in_rsqrt, d_in_rsqrt)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -d_in_rsqrt, d_in_rsqrt)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Do the forward pass."""
        if x.ndim != 3:
            raise ValueError(
                '_NLinear supports only inputs with exactly one batch dimension,'
                ' so `x` must have a shape like (BATCH_SIZE, N_FEATURES, D_EMBEDDING).'
            )
        assert x.shape[-(self.weight.ndim - 1) :] == self.weight.shape[:-1]

        x = x.transpose(0, 1)
        x = x @ self.weight
        x = x.transpose(0, 1)
        if self.bias is not None:
            x = x + self.bias
        return x


class PeriodicEmbeddings(nn.Module):
    """Embeddings for continuous features based on periodic activations.

    See README for details.

    **Shape**

    - Input: `(*, n_features)`
    - Output: `(*, n_features, d_embedding)`

    **Examples**

    >>> batch_size = 2
    >>> n_cont_features = 3
    >>> x = torch.randn(batch_size, n_cont_features)
    >>>
    >>> d_embedding = 24
    >>> m = PeriodicEmbeddings(n_cont_features, d_embedding, lite=False)
    >>> m.get_output_shape()
    torch.Size([3, 24])
    >>> m(x).shape
    torch.Size([2, 3, 24])
    >>>
    >>> m = PeriodicEmbeddings(n_cont_features, d_embedding, lite=True)
    >>> m.get_output_shape()
    torch.Size([3, 24])
    >>> m(x).shape
    torch.Size([2, 3, 24])
    >>>
    >>> # PL embeddings.
    >>> m = PeriodicEmbeddings(n_cont_features, d_embedding=8, activation=False, lite=False)
    >>> m.get_output_shape()
    torch.Size([3, 8])
    >>> m(x).shape
    torch.Size([2, 3, 8])
    """  # noqa: E501

    def __init__(
        self,
        n_features: int,
        d_embedding: int = 24,
        *,
        n_frequencies: int = 48,
        frequency_init_scale: float = 0.01,
        activation: bool = True,
        lite: bool,
    ) -> None:
        """
        Args:
            n_features: the number of features.
            d_embedding: the embedding size.
            n_frequencies: the number of frequencies for each feature.
                (denoted as "k" in Section 3.3 in the paper).
            frequency_init_scale: the initialization scale for the first linear layer
                (denoted as "sigma" in Section 3.3 in the paper).
                **This is an important hyperparameter**, see README for details.
            activation: if `False`, the ReLU activation is not applied.
                Must be `True` if ``lite=True``.
            lite: if True, the outer linear layer is shared between all features.
                See README for details.
        """
        super().__init__()
        self.periodic = _Periodic(n_features, n_frequencies, frequency_init_scale)
        self.linear: Union[nn.Linear, _NLinear]
        if lite:
            # NOTE[DIFF]
            # The lite variation was introduced in a different paper
            # (about the TabR model).
            if not activation:
                raise ValueError('lite=True is allowed only when activation=True')
            self.linear = nn.Linear(2 * n_frequencies, d_embedding)
        else:
            self.linear = _NLinear(n_features, 2 * n_frequencies, d_embedding)
        self.activation = nn.ReLU() if activation else None

    def get_output_shape(self) -> torch.Size:
        """Get the output shape without the batch dimensions."""
        n_features = self.periodic.weight.shape[0]
        d_embedding = (
            self.linear.weight.shape[0]
            if isinstance(self.linear, nn.Linear)
            else self.linear.weight.shape[-1]
        )
        return torch.Size((n_features, d_embedding))

    def forward(self, x: Tensor) -> Tensor:
        """Do the forward pass."""
        x = self.periodic(x)
        x = self.linear(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


def _check_bins(bins: list[Tensor]) -> None:
    if not bins:
        raise ValueError('The list of bins must not be empty')
    for i, feature_bins in enumerate(bins):
        if not isinstance(feature_bins, Tensor):
            raise ValueError(
                'bins must be a list of PyTorch tensors. '
                f'However, for {i=}: {type(bins[i])=}'
            )
        if feature_bins.ndim != 1:
            raise ValueError(
                'Each item of the bin list must have exactly one dimension.'
                f' However, for {i=}: {bins[i].ndim=}'
            )
        if len(feature_bins) < 2:
            raise ValueError(
                'All features must have at least two bin edges.'
                f' However, for {i=}: {len(bins[i])=}'
            )
        if not feature_bins.isfinite().all():
            raise ValueError(
                'Bin edges must not contain nan/inf/-inf.'
                f' However, this is not true for the {i}-th feature'
            )
        if (feature_bins[:-1] >= feature_bins[1:]).any():
            raise ValueError(
                'Bin edges must be sorted.'
                f' However, the for the {i}-th feature, the bin edges are not sorted'
            )
        # Commented out due to spaming warnings.
        # if len(feature_bins) == 2:
        #     warnings.warn(
        #         f'The {i}-th feature has just two bin edges, which means only one bin.'
        #         ' Strictly speaking, using a single bin for the'
        #         ' piecewise-linear encoding should not break anything,'
        #         ' but it is the same as using sklearn.preprocessing.MinMaxScaler'
        #     )


def compute_bins(
    X: torch.Tensor,
    n_bins: int = 48,
    *,
    tree_kwargs: Optional[dict[str, Any]] = None,
    y: Optional[Tensor] = None,
    regression: Optional[bool] = None,
    verbose: bool = False,
) -> list[Tensor]:
    """Compute the bin boundaries for `PiecewiseLinearEncoding` and `PiecewiseLinearEmbeddings`.

    **Usage**

    Compute bins using quantiles (Section 3.2.1 in the paper):

    >>> X_train = torch.randn(10000, 2)
    >>> bins = compute_bins(X_train)

    Compute bins using decision trees (Section 3.2.2 in the paper):

    >>> X_train = torch.randn(10000, 2)
    >>> y_train = torch.randn(len(X_train))
    >>> bins = compute_bins(
    ...     X_train,
    ...     y=y_train,
    ...     regression=True,
    ...     tree_kwargs={'min_samples_leaf': 64, 'min_impurity_decrease': 1e-4},
    ... )

    Args:
        X: the training features.
        n_bins: the number of bins.
        tree_kwargs: keyword arguments for `sklearn.tree.DecisionTreeRegressor`
            (if ``regression=True``) or `sklearn.tree.DecisionTreeClassifier`
            (if ``regression=False``).
            NOTE: requires ``scikit-learn>=1.0,>2`` to be installed.
        y: the training labels (must be provided if ``tree`` is not None).
        regression: whether the labels are regression labels
            (must be provided if ``tree`` is not None).
        verbose: if True and ``tree_kwargs`` is not None, than ``tqdm``
            (must be installed) will report the progress while fitting trees.

    Returns:
        A list of bin edges for all features. For one feature:

        - the maximum possible number of bin edges is ``n_bins + 1``.
        - the minimum possible number of bin edges is ``1``.
    """  # noqa: E501
    if not isinstance(X, Tensor):
        raise ValueError(f'X must be a PyTorch tensor, however: {type(X)=}')
    if X.ndim != 2:
        raise ValueError(f'X must have exactly two dimensions, however: {X.ndim=}')
    if X.shape[0] < 2:
        raise ValueError(f'X must have at least two rows, however: {X.shape[0]=}')
    if X.shape[1] < 1:
        raise ValueError(f'X must have at least one column, however: {X.shape[1]=}')
    if not X.isfinite().all():
        raise ValueError('X must not contain nan/inf/-inf.')
    if (X == X[0]).all(dim=0).any():
        raise ValueError(
            'All columns of X must have at least two distinct values.'
            ' However, X contains columns with just one distinct value.'
        )
    if n_bins <= 1 or n_bins >= len(X):
        raise ValueError(
            'n_bins must be more than 1, but less than len(X), however:'
            f' {n_bins=}, {len(X)=}'
        )

    if tree_kwargs is None:
        if y is not None or regression is not None or verbose:
            raise ValueError(
                'If tree_kwargs is None, then y must be None, regression must be None'
                ' and verbose must be False'
            )

        _upper = 2**24  # 16_777_216
        if len(X) > _upper:
            warnings.warn(
                f'Computing quantile-based bins for more than {_upper} million objects'
                ' may not be possible due to the limitation of PyTorch'
                ' (for details, see https://github.com/pytorch/pytorch/issues/64947;'
                ' if that issue is successfully resolved, this warning may be irrelevant).'  # noqa
                ' As a workaround, subsample the data, i.e. instead of'
                '\ncompute_bins(X, ...)'
                '\ndo'
                '\ncompute_bins(X[torch.randperm(len(X), device=X.device)[:16_777_216]], ...)'  # noqa
                '\nOn CUDA, the computation can still fail with OOM even after'
                ' subsampling. If this is the case, try passing features by groups:'
                '\nbins = sum('
                '\n    compute_bins(X[:, idx], ...)'
                '\n    for idx in torch.arange(len(X), device=X.device).split(group_size),'  # noqa
                '\n    start=[]'
                '\n)'
                '\nAnother option is to perform the computation on CPU:'
                '\ncompute_bins(X.cpu(), ...)'
            )
        del _upper

        # NOTE[DIFF]
        # The code below is more correct than the original implementation,
        # because the original implementation contains an unintentional divergence
        # from what is written in the paper. That divergence affected only the
        # quantile-based embeddings, but not the tree-based embeddings.
        # For historical reference, here is the original, less correct, implementation:
        # https://github.com/yandex-research/tabular-dl-num-embeddings/blob/c1d9eb63c0685b51d7e1bc081cdce6ffdb8886a8/bin/train4.py#L612C30-L612C30
        # (explanation: limiting the number of quantiles by the number of distinct
        #  values is NOT the same as removing identical quantiles after computing them).
        bins = [
            q.unique()
            for q in torch.quantile(
                X, torch.linspace(0.0, 1.0, n_bins + 1).to(X), dim=0
            ).T
        ]
        _check_bins(bins)
        return bins

    else:
        if sklearn_tree is None:
            raise RuntimeError(
                'The scikit-learn package is missing.'
                ' See README.md for installation instructions'
            )
        if y is None or regression is None:
            raise ValueError(
                'If tree_kwargs is not None, then y and regression must not be None'
            )
        if y.ndim != 1:
            raise ValueError(f'y must have exactly one dimension, however: {y.ndim=}')
        if len(y) != len(X):
            raise ValueError(
                f'len(y) must be equal to len(X), however: {len(y)=}, {len(X)=}'
            )
        if y is None or regression is None:
            raise ValueError(
                'If tree_kwargs is not None, then y and regression must not be None'
            )
        if 'max_leaf_nodes' in tree_kwargs:
            raise ValueError(
                'tree_kwargs must not contain the key "max_leaf_nodes"'
                ' (it will be set to n_bins automatically).'
            )

        if verbose:
            if tqdm is None:
                raise ImportError('If verbose is True, tqdm must be installed')
            tqdm_ = tqdm
        else:
            tqdm_ = lambda x: x  # noqa: E731

        if X.device.type != 'cpu' or y.device.type != 'cpu':
            warnings.warn(
                'Computing tree-based bins involves the conversion of the input PyTorch'
                ' tensors to NumPy arrays. The provided PyTorch tensors are not'
                ' located on CPU, so the conversion has some overhead.',
                UserWarning,
            )
        X_numpy = X.cpu().numpy()
        y_numpy = y.cpu().numpy()
        bins = []
        for column in tqdm_(X_numpy.T):
            feature_bin_edges = [float(column.min()), float(column.max())]
            tree = (
                (
                    sklearn_tree.DecisionTreeRegressor
                    if regression
                    else sklearn_tree.DecisionTreeClassifier
                )(max_leaf_nodes=n_bins, **tree_kwargs)
                .fit(column.reshape(-1, 1), y_numpy)
                .tree_
            )
            for node_id in range(tree.node_count):
                # The following condition is True only for split nodes. Source:
                # https://scikit-learn.org/1.0/auto_examples/tree/plot_unveil_tree_structure.html#tree-structure
                if tree.children_left[node_id] != tree.children_right[node_id]:
                    feature_bin_edges.append(float(tree.threshold[node_id]))
            bins.append(torch.as_tensor(feature_bin_edges).unique())
        _check_bins(bins)
        return [x.to(device=X.device, dtype=X.dtype) for x in bins]


class _PiecewiseLinearEncodingImpl(nn.Module):
    """Piecewise-linear encoding.

    NOTE: THIS CLASS SHOULD NOT BE USED DIRECTLY.
    In particular, this class does *not* add any positional information
    to feature encodings. Thus, for Transformer-like models,
    `PiecewiseLinearEmbeddings` is the only valid option.

    Note:
        This is the *encoding* module, not the *embedding* module,
        so it only implements Equation 1 (Figure 1) from the paper,
        and does not have trainable parameters.

    **Shape**

    * Input: ``(*, n_features)``
    * Output: ``(*, n_features, max_n_bins)``,
      where ``max_n_bins`` is the maximum number of bins over all features:
      ``max_n_bins = max(len(b) - 1 for b in bins)``.

    To understand the output structure,
    consider a feature with the number of bins ``n_bins``.
    Formally, its piecewise-linear encoding is a vector of the size ``n_bins``
    that looks as follows::

        x_ple = [1, ..., 1, (x - this_bin_left_edge) / this_bin_width, 0, ..., 0]

    However, this class will instead produce a vector of the size ``max_n_bins``::

        x_ple_actual = [*x_ple[:-1], *zeros(max_n_bins - n_bins), x_ple[-1]]

    In other words:

    * The last encoding component is **always** located in the end,
      even if ``n_bins == 1`` (i.e. even if it is the only component).
    * The leading ``n_bins - 1`` components are located in the beginning.
    * Everything in-between is always set to zeros (like "padding", but in the middle).

    This implementation is *significantly* faster than the original one.
    It relies on two key observations:

    * The piecewise-linear encoding is just
      a non-trainable linear transformation followed by a clamp-based activation.
      Pseudocode: `PiecewiseLinearEncoding(x) = Activation(Linear(x))`.
      The parameters of the linear transformation are defined by the bin edges.
    * Aligning the *last* encoding channel across all features
      allows applying the aforementioned activation simultaneously to all features
      without the loop over features.
    """

    weight: Tensor
    """The weight of the linear transformation mentioned in the class docstring."""

    bias: Tensor
    """The bias of the linear transformation mentioned in the class docstring."""

    single_bin_mask: Optional[Tensor]
    """The indicators of the features with only one bin."""

    mask: Optional[Tensor]
    """The indicators of the "valid" (i.e. "non-padding") part of the encoding."""

    def __init__(self, bins: list[Tensor]) -> None:
        """
        Args:
            bins: the bins computed by `compute_bins`.
        """
        assert len(bins) > 0
        super().__init__()

        n_features = len(bins)
        n_bins = [len(x) - 1 for x in bins]
        max_n_bins = max(n_bins)

        self.register_buffer('weight', torch.zeros(n_features, max_n_bins))
        self.register_buffer('bias', torch.zeros(n_features, max_n_bins))

        single_bin_mask = torch.tensor(n_bins) == 1
        self.register_buffer(
            'single_bin_mask', single_bin_mask if single_bin_mask.any() else None
        )

        self.register_buffer(
            'mask',
            # The mask is needed if features have different number of bins.
            None
            if all(len(x) == len(bins[0]) for x in bins)
            else torch.row_stack(
                [
                    torch.cat(
                        [
                            # The number of bins for this feature, minus 1:
                            torch.ones((len(x) - 1) - 1, dtype=torch.bool),
                            # Unused components (always zeros):
                            torch.zeros(max_n_bins - (len(x) - 1), dtype=torch.bool),
                            # The last bin:
                            torch.ones(1, dtype=torch.bool),
                        ]
                    )
                    # x is a tensor containing the bin bounds for a given feature.
                    for x in bins
                ]
            ),
        )

        for i, bin_edges in enumerate(bins):
            # Formally, the piecewise-linear encoding of one feature looks as follows:
            # `[1, ..., 1, (x - this_bin_left_edge) / this_bin_width, 0, ..., 0]`
            # The linear transformation based on the weight and bias defined below
            # implements the expression in the middle before the clipping to [0, 1].
            # Note that the actual encoding layout produced by this class
            # is slightly different. See the docstring of this class for details.
            bin_width = bin_edges.diff()
            w = 1.0 / bin_width
            b = -bin_edges[:-1] / bin_width
            # The last encoding component:
            self.weight[i, -1] = w[-1]
            self.bias[i, -1] = b[-1]
            # The leading encoding components:
            self.weight[i, : n_bins[i] - 1] = w[:-1]
            self.bias[i, : n_bins[i] - 1] = b[:-1]
            # All in-between components will always be zeros,
            # because the weight and bias are initialized with zeros.

    def get_max_n_bins(self) -> int:
        return self.weight.shape[-1]

    def forward(self, x: Tensor) -> Tensor:
        """Do the forward pass."""
        x = torch.addcmul(self.bias, self.weight, x[..., None])
        if x.shape[-1] > 1:
            x = torch.cat(
                [
                    x[..., :1].clamp_max(1.0),
                    x[..., 1:-1].clamp(0.0, 1.0),
                    (
                        x[..., -1:].clamp_min(0.0)
                        if self.single_bin_mask is None
                        else torch.where(
                            # For features with only one bin,
                            # the whole "piecewise-linear" encoding effectively behaves
                            # like mix-max scaling
                            # (assuming that the edges of the single bin
                            #  are the minimum and maximum feature values).
                            self.single_bin_mask[..., None],
                            x[..., -1:],
                            x[..., -1:].clamp_min(0.0),
                        )
                    ),
                ],
                dim=-1,
            )
        return x


class PiecewiseLinearEncoding(nn.Module):
    """Piecewise-linear encoding.

    See README for detailed explanation.

    **Shape**

    - Input: ``(*, n_features)``
    - Output: ``(*, total_n_bins)``,
      where ``total_n_bins`` is the total number of bins for all features:
      ``total_n_bins = sum(len(b) - 1 for b in bins)``.

    Technically, the output of this module is the flattened output
    of `_PiecewiseLinearEncoding` with all "padding" values removed.
    """

    def __init__(self, bins: list[Tensor]) -> None:
        """
        Args:
            bins: the bins computed by `compute_bins`.
        """
        super().__init__()
        self.impl = _PiecewiseLinearEncodingImpl(bins)

    def get_output_shape(self) -> torch.Size:
        """Get the output shape without the batch dimensions."""
        total_n_bins = (
            self.impl.weight.shape.numel()
            if self.impl.mask is None
            else int(self.impl.mask.long().sum().cpu().item())
        )
        return torch.Size((total_n_bins,))

    def forward(self, x: Tensor) -> Tensor:
        """Do the forward pass."""
        x = self.impl(x)
        return x.flatten(-2) if self.impl.mask is None else x[:, self.impl.mask]


class PiecewiseLinearEmbeddings(nn.Module):
    """Piecewise-linear embeddings.

    **Shape**

    - Input: ``(batch_size, n_features)``
    - Output: ``(batch_size, n_features, d_embedding)``
    """

    def __init__(
        self,
        bins: list[Tensor],
        d_embedding: int,
        *,
        activation: bool,
        version: Literal[None, 'A', 'B'] = None,
    ) -> None:
        """
        Args:
            bins: the bins computed by `compute_bins`.
            d_embedding: the embedding size.
            activation: if True, the ReLU activation is additionally applied in the end.
            version: the preset for various implementation details, such as
                parametrization and initialization. See README for details.
        """
        if d_embedding <= 0:
            raise ValueError(
                f'd_embedding must be a positive integer, however: {d_embedding=}'
            )
        _check_bins(bins)
        if version is None:
            warnings.warn(
                'The `version` argument is not provided, so version="A" will be used'
                ' for backward compatibility.'
                ' See README for recommendations regarding `version`.'
                ' In future, omitting this argument will result in an exception.'
            )
            version = 'A'

        super().__init__()
        n_features = len(bins)
        # NOTE[DIFF]
        # version="B" was introduced in a different paper (about the TabM model).
        is_version_B = version == 'B'

        self.linear0 = (
            LinearEmbeddings(n_features, d_embedding) if is_version_B else None
        )
        self.impl = _PiecewiseLinearEncodingImpl(bins)
        self.linear = _NLinear(
            len(bins),
            self.impl.get_max_n_bins(),
            d_embedding,
            # For the version "B", the bias is already presented in self.linear0.
            bias=not is_version_B,
        )
        if is_version_B:
            # Because of the following line, at initialization,
            # the whole embedding behaves like a linear embedding.
            # The piecewise-linear component is incrementally learnt during training.
            nn.init.zeros_(self.linear.weight)
        self.activation = nn.ReLU() if activation else None

    def get_output_shape(self) -> torch.Size:
        """Get the output shape without the batch dimensions."""
        n_features = self.linear.weight.shape[0]
        d_embedding = self.linear.weight.shape[2]
        return torch.Size((n_features, d_embedding))

    def forward(self, x: Tensor) -> Tensor:
        """Do the forward pass."""
        if x.ndim != 2:
            raise ValueError(
                'For now, only inputs with exactly one batch dimension are supported.'
            )

        x_linear = None if self.linear0 is None else self.linear0(x)

        x_ple = self.impl(x)
        x_ple = self.linear(x_ple)
        if self.activation is not None:
            x_ple = self.activation(x_ple)
        return x_ple if x_linear is None else x_linear + x_ple