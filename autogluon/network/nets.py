import ConfigSpace as CS
from gluoncv.model_zoo import get_model

from ..core import *
from ..space import *
#from .utils import Net, autogluon_nets

#__all__ = ['Nets']

def get_finetune_network(model_name, num_classes, ctx):
    finetune_net = get_model(model_name, pretrained=True)
    # change the last fully connected layer to match the number of classes
    with finetune_net.name_scope():
        finetune_net.output = gluon.nn.Dense(num_classes)
    # initialize and context
    finetune_net.output.initialize(init.Xavier(), ctx=ctx)
    finetune_net.collect_params().reset_ctx(ctx)
    finetune_net.hybridize()
    return finetune_net
