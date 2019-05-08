backend = 'mxnet'
try:
    import mxnet as mx
except ImportError:
    try:
        import torch
        backend = 'torch'
    except ImportError:
        raise RuntimeError('AutoGluon depends on MxNet or PyTorch, '
                           'please install at least one backend.')
