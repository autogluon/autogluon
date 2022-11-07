import pytorch_lightning as pl

import autogluon.timeseries as agts


def set_random_seed(seed: int) -> None:
    """Set the seed for Numpy, PyTorch and MXNet random number generators."""
    if seed is not None:
        pl.seed_everything(seed)
        if agts.MXNET_INSTALLED:
            import mxnet as mx

            mx.random.seed(seed)
