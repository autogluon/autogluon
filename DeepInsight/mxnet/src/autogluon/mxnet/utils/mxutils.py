"""MXNet Utils Functions"""

import os
import math
import mxnet as mx

mx_ver = mx.__version__[:3]

__all__ = [
    'update_params',
    'collect_params',
    'get_data_rec',
    'read_remote_ips']


def update_params(net, params, multi_precision=False, ctx=mx.cpu(0)):
    param_dict = net._collect_params_with_prefix()
    kwargs = {'ctx': None} if mx_ver == '1.4' else {'cast_dtype': multi_precision, 'ctx': None}
    for k, v in param_dict.items():
        param_dict[k]._load_init(params[k], **kwargs)


def collect_params(net):
    params = net._collect_params_with_prefix()
    param_dict = {key : val._reduce() for key, val in params.items()}
    return param_dict


def get_data_rec(input_size, crop_ratio,
                 rec_file, rec_file_idx,
                 batch_size, num_workers, train=True,
                 shuffle=True,
                 jitter_param=0.4, max_rotate_angle=0):
    import mxnet as mx
    from mxnet import gluon
    rec_file = os.path.expanduser(rec_file)
    rec_file_idx = os.path.expanduser(rec_file_idx)
    lighting_param = 0.1
    input_size = input_size
    crop_ratio = crop_ratio if crop_ratio > 0 else 0.875
    resize = int(math.ceil(input_size / crop_ratio))
    mean_rgb = [123.68, 116.779, 103.939]
    std_rgb = [58.393, 57.12, 57.375]

    if train:
        data_loader = mx.io.ImageRecordIter(
            path_imgrec         = rec_file,
            path_imgidx         = rec_file_idx,
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
    else:
        data_loader = mx.io.ImageRecordIter(
            path_imgrec         = rec_file,
            path_imgidx         = rec_file_idx,
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
    return data_loader


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

