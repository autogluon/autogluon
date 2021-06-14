"""ag.Space sanitizer for certain hyperparameters"""
import warnings
import numpy as np
from autogluon.core import Categorical, Int


def sanitize_batch_size(batch_size, min_value=1, max_value=np.inf):
    if isinstance(batch_size, Categorical):
        valid_bs = []
        bs_values = batch_size.data
        for bs_value in bs_values:
            if isinstance(bs_value, int) and min_value < bs_value < max_value:
                valid_bs.append(bs_value)
        if valid_bs != bs_values:
            warnings.warn(f'Pruning batch size from {batch_size} to {valid_bs} due to memory limit.')
        if len(valid_bs) == 1:
            new_bs = valid_bs[0]
        else:
            new_bs = Categorical(*valid_bs)
    elif isinstance(batch_size, Int):
        lower = batch_size.lower
        upper = batch_size.upper
        if not isinstance(lower, int) or not isinstance(upper, int):
            raise TypeError(f'Invalid lower {lower} or upper {upper} bound for Int space')
        lower = max(lower, min_value)
        upper = min(upper, max_value)
        new_bs = Int(lower=lower, upper=upper)
        if lower != batch_size.lower or upper != batch_size.higher:
            warnings.warn(f'Adjusting batch size range from {batch_size} to {new_bs} due to memory limit.')
    elif isinstance(batch_size, int):
        new_bs = max(min(batch_size, max_value), min_value)
        if new_bs != batch_size:
            warnings.warn(f'Adjusting batch size from {batch_size} to {new_bs} due to memory limit.')
    else:
        raise TypeError(f'Expecting batch size to be (Categorical/Int/int), given {type(batch_size)}.')
    return new_bs
