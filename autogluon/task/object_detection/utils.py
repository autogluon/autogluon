from .nets import *
import pdb

def get_network(net, transfer_classes, ctx):
    if type(net) == str:
        net = get_built_in_network(net, transfer_classes, ctx=ctx)
    else:
        net.initialize(ctx=ctx)
    return net