__all__ = ['Losses']


class Losses(object):
    def __init__(self, loss_list):
        assert isinstance(loss_list, list), type(loss_list)
        self.loss_list = loss_list
        self.search_space = None
        self.add_search_space()

    def add_search_space(self):
        pass
