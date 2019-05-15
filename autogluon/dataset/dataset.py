__all__ = ['Dataset']


class Dataset(object):
    def __init__(self, train_path=None, val_path=None):
        # TODO (cgraywang): add search space, handle batch_size, num_workers
        self.train_path = train_path
        self.val_path = val_path
        self._read_dataset()
        self.search_space = None
        self.add_search_space()
        self.train_data = None
        self.val_data = None

    def _read_dataset(self):
        pass

    def _set_search_space(self, cs):
        self.search_space = cs

    def add_search_space(self):
        pass

    def get_search_space(self):
        return self.search_space
