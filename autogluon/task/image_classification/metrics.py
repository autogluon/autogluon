import mxnet
import gluoncv

#from ...metric import autogluon_metrics, Metric

__all__ = ['get_metric_instance']

metrics = {'accuracy': mxnet.metric.Accuracy,
           'topkaccuracy': mxnet.metric.TopKAccuracy,
           'f1': mxnet.metric.F1,
           'mcc': mxnet.metric.MCC,
           'perplexity': mxnet.metric.Perplexity,
           'mae': mxnet.metric.MAE,
           'mse': mxnet.metric.MSE,
           'rmse': mxnet.metric.RMSE,
           'crossentropy': mxnet.metric.CrossEntropy,
           'pearsoncorrelation': mxnet.metric.PearsonCorrelation}

def get_metric_instance(name, **kwargs):
    """Returns a metric instance by name

    Parameters
    ----------
    name : str
        Name of the metric.

    Returns
    -------
    metric
        The metric instance.
    """
    name = name.lower()
    if name not in metrics:
        err_str = '"%s" is not among the following metric list:\n\t' % (name)
        err_str += '%s' % ('\n\t'.join(sorted(metrics.keys())))
        raise ValueError(err_str)
    metric = metrics[name]()
    return metric
