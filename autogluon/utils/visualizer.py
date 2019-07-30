import numpy as np
from matplotlib import pyplot

__all__ = ['Visualizer']


class Visualizer(object):
    def __init__(self):
        pass

    @staticmethod
    def visualize_dataset_label_histogram(a, b):
        min_len = min(len(a._label), len(b._label))
        pyplot.hist([a._label[:min_len], b._label[:min_len]],
                    bins=len(np.unique(a._label)),
                    label=['a', 'b'])
        pyplot.legend(loc='upper right')
        pyplot.savefig('./histogram.png')
