# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Initializer in GluonNLP."""
__all__ = ['TruncNorm']

import warnings
import math
import mxnet as mx
from mxnet.initializer import Initializer


def norm_cdf(x):
    return (1. + math.erf(x / math.sqrt(2.))) / 2.


@mx.initializer.register
class TruncNorm(Initializer):
    r"""Initialize the weight by drawing sample from truncated normal distribution with
    provided mean and standard deviation. Values whose magnitude is more than 2 standard deviations
    from the mean are dropped and re-picked.

    In the implementation, we used the method described in
    https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf, which is also
    obtained in PyTorch.

    Parameters
    ----------
    mean
        Mean of the underlying normal distribution
    stdev
        Standard deviation of the underlying normal distribution
    scale
        The scale of the truncated distribution.
        The values
    **kwargs
        Additional parameters for base Initializer.
    """
    def __init__(self, mean: float = 0, stdev: float = 0.01,
                 scale=2, **kwargs):
        super(TruncNorm, self).__init__(**kwargs)
        self._mean = mean
        self._stdev = stdev
        self._scale = scale
        self._a = mean - scale * stdev
        self._b = mean + scale * stdev
        if (mean < self._a - 2 * stdev) or (mean > self._b + 2 * stdev):
            warnings.warn("mean is more than 2 std from [a, b] in init.TruncNorm. "
                          "The distribution of values may be incorrect.",
                          stacklevel=2)
        self._l = norm_cdf(-scale)
        self._u = norm_cdf(scale)

    def _init_weight(self, name, arr):
        # pylint: disable=unused-argument
        """Abstract method to Initialize weight."""
        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        arr[:] = mx.np.random.uniform(2 * self._l - 1, 2 * self._u - 1, size=arr.shape, ctx=arr.ctx)
        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        arr[:] = mx.npx.erfinv(arr)

        # Transform to proper mean, std
        arr *= self._stdev * math.sqrt(2.)
        arr += self._mean
        # Clamp to ensure it's in the proper range
        arr[:] = arr.clip(self._a, self._b)
