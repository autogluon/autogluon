"""MXNet Utils Functions"""

__all__ = ['update_params']

def update_params(net, params):
    param_dict = net.collect_params()
    for k, v in param_dict.items():
        param_dict[k].set_data(params[k].data())
