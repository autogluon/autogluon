import numpy as np

import jax.numpy as jax_np

def clip_for_log(X):
    eps = np.finfo(X.dtype).tiny
    return np.clip(X, eps, 1-eps)


def clip(X):
    eps = np.finfo(X.dtype).tiny
    return np.clip(X, eps, 1-eps)

def clip_jax(X):
    eps = jax_np.finfo(X.dtype).eps
    return jax_np.clip(X, eps, 1-eps)