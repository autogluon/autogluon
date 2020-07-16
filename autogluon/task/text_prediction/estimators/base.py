import abc


class BaseEstimator(abc.ABC):
    def __init__(self, config=None, logger=None, reporter=None):
        super().__init__()
        if config is None:
            self._config = self.__class__.get_cfg()
        else:
            base_config = self.__class__.get_cfg()
            self._config = base_config.clone_merge(config)
        self._logger = logger
        self._reporter = reporter

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
