"""MXNet Utils Functions"""

import os
import math

__all__ = ['update_params', 'get_data_rec', 'read_remote_ips']

def update_params(net, params):
    param_dict = net.collect_params()
    for k, v in param_dict.items():
        param_dict[k].set_data(params[k].data())


def get_data_rec(input_size, crop_ratio, rec_train, rec_train_idx,
                 rec_val, rec_val_idx, batch_size, num_workers,
                 jitter_param, max_rotate_angle):
    import mxnet as mx
    from mxnet import gluon
    rec_train = os.path.expanduser(rec_train)
    rec_train_idx = os.path.expanduser(rec_train_idx)
    rec_val = os.path.expanduser(rec_val)
    rec_val_idx = os.path.expanduser(rec_val_idx)
    
    lighting_param = 0.1
    input_size = input_size
    crop_ratio = crop_ratio if crop_ratio > 0 else 0.875
    resize = int(math.ceil(input_size / crop_ratio))
    mean_rgb = [123.68, 116.779, 103.939]
    std_rgb = [58.393, 57.12, 57.375]

    def batch_fn(batch, ctx):
        data = gluon.utils.split_and_load(batch.data[0], ctx_list=ctx, batch_axis=0)
        label = gluon.utils.split_and_load(batch.label[0], ctx_list=ctx, batch_axis=0)
        return data, label

    print('rec_train', rec_train)
    print('rec_train_idx', rec_train_idx)
    train_data = mx.io.ImageRecordIter(
        path_imgrec         = rec_train,
        path_imgidx         = rec_train_idx,
        preprocess_threads  = num_workers,
        shuffle             = True,
        batch_size          = batch_size,

        data_shape          = (3, input_size, input_size),
        mean_r              = mean_rgb[0],
        mean_g              = mean_rgb[1],
        mean_b              = mean_rgb[2],
        std_r               = std_rgb[0],
        std_g               = std_rgb[1],
        std_b               = std_rgb[2],
        rand_mirror         = True,
        random_resized_crop = True,
        max_aspect_ratio    = 4. / 3.,
        min_aspect_ratio    = 3. / 4.,
        max_random_area     = 1,
        min_random_area     = 0.08,
        max_rotate_angle    = max_rotate_angle,
        brightness          = jitter_param,
        saturation          = jitter_param,
        contrast            = jitter_param,
        pca_noise           = lighting_param,
    )
    # val_input_size = 320
    val_data = mx.io.ImageRecordIter(
        path_imgrec         = rec_val,
        path_imgidx         = rec_val_idx,
        preprocess_threads  = num_workers,
        shuffle             = False,
        batch_size          = batch_size,

        resize              = resize,
        data_shape          = (3, input_size, input_size),
        mean_r              = mean_rgb[0],
        mean_g              = mean_rgb[1],
        mean_b              = mean_rgb[2],
        std_r               = std_rgb[0],
        std_g               = std_rgb[1],
        std_b               = std_rgb[2],
    )
    return train_data, val_data, batch_fn

def read_remote_ips(filename):
    ip_addrs = []
    if filename is None:
        return ip_addrs
    with open("remote_ips.txt", "r") as myfile:
        line = myfile.readline()
        while line != '':
            ip_addrs.append(line.rstrip())
            line = myfile.readline()
    return ip_addrs

