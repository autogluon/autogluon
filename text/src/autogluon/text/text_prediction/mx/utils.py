import mxnet as mx


def average_checkpoints(net, checkpoint_paths, out_path):
    for path in checkpoint_paths:

