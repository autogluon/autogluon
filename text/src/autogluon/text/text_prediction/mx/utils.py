import mxnet as mx
from mxnet.util import use_np

@use_np
def average_checkpoints(net, checkpoint_paths, out_path):
    for path in checkpoint_paths:
        pass
