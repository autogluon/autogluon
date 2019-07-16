import mxnet
import gluoncv

from autogluon.metric import autogluon_metrics, Metric

__all__ = ['get_metric', 'get_metric_instance']

#TODO(cgraywang): abstract general metric shared across tasks
metrics = {'VOC07MApMetric': gluoncv.utils.metrics.voc_detection.VOC07MApMetric}


@autogluon_metrics
def get_metric(name, **kwargs):
    """Returns a metric with search space by name

    Parameters
    ----------
    name : str
        Name of the metric.

    Returns
    -------
    metric
        The metric with search space.
    """
    if name not in metrics and name.lower() not in metrics:
        err_str = '"%s" is not among the following metric list:\n\t' % (name)
        err_str += '%s' % ('\n\t'.join(sorted(metrics.keys())))
        raise ValueError(err_str)
    metric = Metric(name)
    return metric


def get_metric_instance(name, *args):
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
    if name not in metrics and name.lower() not in metrics:
        err_str = '"%s" is not among the following metric list:\n\t' % (name)
        err_str += '%s' % ('\n\t'.join(sorted(metrics.keys())))
        raise ValueError(err_str)
    metric = metrics[name](*args)
    return metric


@autogluon_metrics
def VOC07MApMetric(**kwargs):
    pass
