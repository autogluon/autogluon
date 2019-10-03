import mxnet as mx
import numpy as np
from matplotlib import pyplot
try:
    import graphviz
except ImportError:
    graphviz = None

__all__ = ['plot_network']

def plot_network(block, shape=(1, 3, 224, 224), savefile=False):
    """Plot network to visualize internal structures.

    Parameters
    ----------
    block (mxnet.gluon.HybridBlock): mxnet.gluon.HybridBlock
        A hybridizable network to be visualized.
    shape (tuple of int):
        Desired input shape, default is (1, 3, 224, 224).
    save_prefix (str or None):
        If not `None`, will save rendered pdf to disk with prefix.

    """
    if graphviz is None:
        raise RuntimeError("Cannot import graphviz.")
    if not isinstance(block, mx.gluon.HybridBlock):
        raise ValueError("block must be HybridBlock, given {}".format(type(block)))
    data = mx.sym.var('data')
    sym = block(data)
    if isinstance(sym, tuple):
        sym = mx.sym.Group(sym)

    a = mx.viz.plot_network(sym, shape={'data':shape},
                            node_attrs={'shape':'rect', 'fixedsize':'false'})
    if savefile:
        a.view(savefile)
    return a
