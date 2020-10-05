import mxnet

__all__ = ['get_metric_instance']

metrics = {'accuracy': mxnet.metric.Accuracy,
           'topkaccuracy': mxnet.metric.TopKAccuracy,
           'mae': mxnet.metric.MAE,
           'mse': mxnet.metric.MSE,
           'rmse': mxnet.metric.RMSE,
           'crossentropy': mxnet.metric.CrossEntropy}

def get_metric_instance(name, **kwargs):
    """Returns a metric instance by name

    Args:
        name : str
            Name of the metric.
    """
    name = name.lower()
    if name not in metrics:
        err_str = '"%s" is not among the following metric list:\n\t' % (name)
        err_str += '%s' % ('\n\t'.join(sorted(metrics.keys())))
        raise ValueError(err_str)
    metric = metrics[name]()
    return metric
