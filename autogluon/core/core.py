from abc import ABC, abstractmethod

__all__ = ['fit', 'BaseAutoObject']


class BaseAutoObject(ABC):
    def __init__(self):
        super(BaseAutoObject, self).__init__()
        self._search_space = None

    @property
    def search_space(self):
        return self._search_space

    @search_space.setter
    def search_space(self, cs):
        self._search_space = cs

    @abstractmethod
    def _add_search_space(self):
        pass

    @abstractmethod
    def _get_search_space_strs(self):
        pass



# TODO (cgraywang): put into class that can be inherited and add readme
# This is an abstract method
def fit(data=None,
        nets=None,
        optimizers=None,
        metrics=None,
        losses=None,
        searcher=None,
        trial_scheduler=None,
        resume=False,
        savedir='./outputdir/',
        visualizer='tensorboard',
        stop_criterion={},
        resources_per_trial={},
        backend='default',
        **kwargs):
    r"""
    Abstract Fit networks on dataset

    Parameters
    ----------
    data: Input data. It could be:
        autogluon.Datasets
        task.Datasets
    nets: autogluon.Nets
    optimizers: autogluon.Optimizers
    metrics: autogluon.Metrics
    losses: autogluon.Losses
    stop_criterion (dict): The stopping criteria. The keys may be any field in
        the return result of 'train()', whichever is reached first.
        Defaults to empty dict.
    resources_per_trial (dict): Machine resources to allocate per trial,
        e.g. ``{"max_num_cpus": 64, "max_num_gpus": 8}``. Note that GPUs will not be
        assigned unless you specify them here.
    savedir (str): Local dir to save training results to.
    searcher: Search Algorithm.
    trial_scheduler: Scheduler for executing
        the experiment. Choose among FIFO (default) and HyperBand.
    resume (bool): If checkpoint exists, the experiment will
        resume from there.
    backend: support autogluon default backend, ray. (Will support SageMaker)
    **kwargs: Used for backwards compatibility.

    Returns
    ----------
    model: the parameters associated with the best model. (TODO: use trial to infer for now)
    best_result: accuracy
    best_config: best configuration
    """

    raise NotImplementedError("The method not implemented")
