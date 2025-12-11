import numpy as np
from numba import njit, prange


@njit(parallel=True, fastmath=True)
def eval_order_fused(X_base: np.ndarray, idx_mat: np.ndarray, op_mat: np.ndarray) -> np.ndarray:
    n_rows, n_base = X_base.shape
    n_feats, order = idx_mat.shape

    out = np.empty((n_rows, n_feats), dtype=X_base.dtype)

    for i in prange(n_rows):
        for f in range(n_feats):
            idx_row = idx_mat[f]  # 1D view: length = order
            ops_row = op_mat[f]  # 1D view: length = order-1

            v = X_base[i, idx_row[0]]

            for k in range(1, order):
                b = X_base[i, idx_row[k]]
                op = ops_row[k - 1]

                if op == 0:  # +
                    v += b
                elif op == 1:  # -
                    v -= b
                elif op == 2:  # *
                    v *= b
                else:  # /
                    if b == 0.0:
                        v = np.nan
                    else:
                        v /= b

            out[i, f] = v

    return out
