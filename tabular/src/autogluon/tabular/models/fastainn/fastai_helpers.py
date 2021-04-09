import pickle
import warnings
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from fastai.data.transforms import get_c
from fastai.layers import Embedding, LinBnDrop, SigmoidRange
from fastai.learner import Learner
from fastai.tabular.learner import TabularLearner
from fastai.tabular.model import get_emb_sz, tabular_config
from fastai.torch_core import flatten_check, Module
from fastcore.basics import ifnone
from fastcore.meta import delegates
from fastcore.xtras import is_listy


class TabularModel(Module):
    """Basic model for tabular data."""

    def __init__(self, emb_szs, n_cont, out_sz, layers, ps=None, embed_p=0.,
                 y_range=None, use_bn=True, bn_final=False, bn_cont=True, act_cls=nn.ReLU(inplace=True)):
        ps = ifnone(ps, [0] * len(layers))
        if not is_listy(ps): ps = [ps] * len(layers)
        self.embeds = nn.ModuleList([Embedding(ni, nf) for ni, nf in emb_szs])
        self.emb_drop = nn.Dropout(embed_p)
        self.bn_cont = nn.BatchNorm1d(n_cont) if bn_cont else None
        n_emb = sum(e.embedding_dim for e in self.embeds)
        self.n_emb, self.n_cont = n_emb, n_cont
        sizes = [n_emb + n_cont] + layers + [out_sz]
        actns = [act_cls for _ in range(len(sizes) - 2)] + [None]
        _layers = [LinBnDrop(sizes[i], sizes[i + 1], bn=use_bn and (i != len(actns) - 1 or bn_final), p=p, act=a, lin_first=True)
                   for i, (p, a) in enumerate(zip(ps + [0.], actns))]
        if y_range is not None: _layers.append(SigmoidRange(*y_range))
        self.layers = nn.Sequential(*_layers)

    def forward(self, x_cat, x_cont=None):
        if self.n_emb != 0:
            x = [e(x_cat[:, i]) for i, e in enumerate(self.embeds)]
            x = torch.cat(x, 1)
            x = self.emb_drop(x)
        if self.n_cont != 0:
            if self.bn_cont is not None: x_cont = self.bn_cont(x_cont)
            x = torch.cat([x, x_cont], 1) if self.n_emb != 0 else x_cont
        return self.layers(x)


@delegates(Learner.__init__)
def tabular_learner(dls, layers=None, emb_szs=None, config=None, n_out=None, y_range=None, **kwargs):
    "Get a `Learner` using `dls`, with `metrics`, including a `TabularModel` created using the remaining params."
    if config is None: config = tabular_config()
    if layers is None: layers = [200, 100]
    to = dls.train_ds
    emb_szs = get_emb_sz(dls.train_ds, {} if emb_szs is None else emb_szs)
    if n_out is None: n_out = get_c(dls)
    assert n_out, "`n_out` is not defined, and could not be inferred from data, set `dls.c` or pass `n_out`"
    if y_range is None and 'y_range' in config: y_range = config.pop('y_range')
    model = TabularModel(emb_szs, len(dls.cont_names), n_out, layers, y_range=y_range, **config)
    return TabularLearner(dls, model, **kwargs)


def export(model, filename_or_stream='export.pkl', pickle_module=pickle, pickle_protocol=2):
    from fastai.torch_core import rank_distrib
    import torch
    "Export the content of `self` without the items and the optimizer state for inference"
    if rank_distrib(): return  # don't export if child proc
    model._end_cleanup()
    old_dbunch = model.dls
    model.dls = model.dls.new_empty()
    state = model.opt.state_dict() if model.opt is not None else None
    model.opt = None
    target = open(model.path / filename_or_stream, 'wb') if is_pathlike(filename_or_stream) else filename_or_stream
    with warnings.catch_warnings():
        # To avoid the warning that come from PyTorch about model not being checked
        warnings.simplefilter("ignore")
        torch.save(model, target, pickle_module=pickle_module, pickle_protocol=pickle_protocol)
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
