from typing import AnyStr

__all__ = ['Initializers']


class Initializers(object):
    def __init__(self, initializer_list):
        assert isinstance(initializer_list, list), type(initializer_list)
        self.initializer_list = initializer_list
        self._search_space = None
        # self.add_search_space()
        # TODO (ghaipiyu): Parse user provided config to generate initializers
        
    @property
    def search_space(self):
        return self._search_space
