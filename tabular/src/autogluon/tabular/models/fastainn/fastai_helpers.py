import pickle
import warnings
from pathlib import Path
from typing import Any

import torch
from fastai.torch_core import flatten_check


def export(model, filename_or_stream="export.pkl", pickle_module=pickle, pickle_protocol=2):
    import torch
    from fastai.torch_core import rank_distrib

    "Export the content of `self` without the items and the optimizer state for inference"
    if rank_distrib():
        return  # don't export if child proc
    model._end_cleanup()
    old_dbunch = model.dls
    model.dls = model.dls.new_empty()
    state = model.opt.state_dict() if model.opt is not None else None
    model.opt = None
    target = open(model.path / filename_or_stream, "wb") if is_pathlike(filename_or_stream) else filename_or_stream
    with warnings.catch_warnings():
        # To avoid the warning that come from PyTorch about model not being checked
        warnings.simplefilter("ignore")
        torch.save(model, target, pickle_module=pickle_module, pickle_protocol=pickle_protocol)  # nosec B614
    model.create_opt()
    if state is not None:
        model.opt.load_state_dict(state)
    model.dls = old_dbunch


def is_pathlike(x: Any) -> bool:
    return isinstance(x, (str, Path))


def medae(inp, targ):
    "Mean absolute error between `inp` and `targ`."
    inp, targ = flatten_check(inp, targ)
    e = torch.abs(inp - targ)
    return torch.median(e).item()
