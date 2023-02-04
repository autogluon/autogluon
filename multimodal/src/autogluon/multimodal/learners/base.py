from abc import ABC, abstractmethod

class BaseLearner(ABC):

    @abstractmethod
    def predict(self, *args, **kwargs):
        pass

    @abstractmethod
    def fit(self, *args, **kwargs):
        pass

    @abstractmethod
    def evaluate(self, *args, **kwargs):
        pass

    @abstractmethod
    def extract_embedding(self, *args, **kwargs):
        pass

    @abstractmethod
    def save(self):
        pass

    @abstractmethod
    def load(self):
        pass