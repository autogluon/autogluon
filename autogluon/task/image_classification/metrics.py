import mxnet
import gluoncv

from autogluon.metric import autogluon_metrics, Metric

__all__ = ['get_metric', 'get_metric_instance']

metrics = {'Accuracy': mxnet.metric.Accuracy,
           'TopKAccuracy': mxnet.metric.TopKAccuracy,
           'F1': mxnet.metric.F1,
           'MCC': mxnet.metric.MCC,
           'Perplexity': mxnet.metric.Perplexity,
           'MAE': mxnet.metric.MAE,
           'MSE': mxnet.metric.MSE,
           'RMSE': mxnet.metric.RMSE,
           'CrossEntropy': mxnet.metric.CrossEntropy,
           'PearsonCorrelation': mxnet.metric.PearsonCorrelation}


@autogluon_metrics
def get_metric(name, **kwargs):
    """Returns a metric with search space by name

    Parameters
    ----------
    name : str
        Name of the model.

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


def get_metric_instance(name, **kwargs):
    """Returns a metric with search space by name

    Parameters
    ----------
    name : str
        Name of the model.

    Returns
    -------
    metric
        The metric with search space.
    """
    if name not in metrics and name.lower() not in metrics:
        err_str = '"%s" is not among the following metric list:\n\t' % (name)
        err_str += '%s' % ('\n\t'.join(sorted(metrics.keys())))
        raise ValueError(err_str)
    metric = metrics[name]()
    return metric


@autogluon_metrics
def Accuracy(**kwargs):
    pass


@autogluon_metrics
def TopKAccuracy(**kwargs):
    pass


@autogluon_metrics
def F1(**kwargs):
    pass


@autogluon_metrics
def MCC(**kwargs):
    pass


@autogluon_metrics
def Perplexity(**kwargs):
    pass


@autogluon_metrics
def MAE(**kwargs):
    pass


@autogluon_metrics
def MSE(**kwargs):
    pass


@autogluon_metrics
def RMSE(**kwargs):
    pass


@autogluon_metrics
def CrossEntropy(**kwargs):
    pass


@autogluon_metrics
def PearsonCorrelation(**kwargs):
    pass