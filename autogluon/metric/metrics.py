__all__ = ['Metrics']


class Metrics(object):
    def __init__(self, metric_list):
        assert isinstance(metric_list, list), type(metric_list)
        self.metric_list = metric_list
        self.search_space = None
        self.add_search_space()

    def add_search_space(self):
        pass

