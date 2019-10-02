from abc import ABC, abstractmethod

__all__ = ['BaseAutoObject']


class BaseAutoObject(ABC):
    """The abstract auto object containing the search space.
    """
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
