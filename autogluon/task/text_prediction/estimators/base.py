import abc


class BaseEstimator(abc.ABC):
    def __init__(self, config=None, logger=None):
        """

        Parameters
        ----------
        base_config
            The basic configuration of the estimator
        search_space
            The search space. Here, we may just specify part of the
        logger
        """
        super().__init__()
        if base_config is None:
            self._config = self.__class__.get_cfg()
        else:
            base_config = self.__class__.get_cfg()
            self._config = base_config.clone_merge(config)
        self._logger = logger

    @property
    def config(self):
        return self._config

    @staticmethod
    @abc.abstractmethod
    def get_cfg(key=None):
        pass

    @abc.abstractmethod
    def fit(self, train_data, **kwargs):
        pass

    @abc.abstractmethod
    def predict(self, test_data):
        pass

    @abc.abstractmethod
    def save(self, dir_path):
        pass

    @classmethod
    @abc.abstractmethod
    def load(cls, dir_path):
        pass
