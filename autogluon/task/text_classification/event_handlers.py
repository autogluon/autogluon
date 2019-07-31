from typing import AnyStr

import numpy as np
from mxnet import gluon

from autogluon.estimator.event_handler import DataLoaderHandler

__all__ = ['TextDataLoaderHandler']


class TextDataLoaderHandler(DataLoaderHandler):
    """
    Data Loader handler which is passed to the Estimator.
    This controls the logic for how the data is loaded and passed as an input to the model.
    """

    def __init__(self, model_name: AnyStr = None):
        # Need model name here to differentiate how to load the data from the dataloader
        self.model_name = model_name

    def batch_begin(self, estimator, *args, **kwargs):
        """
        :param estimator:
        :param batch: The batch of data
        :param ctx: The context in which to load the data.
        :param batch_axis: The batch axis about which to split the data onto multiple devices if context is passed as a list
        :return: A tuple of : (data, length), label and batch_size
        """

        if 'bert' in self.model_name:
            return self.bert_data_loader(**kwargs)

        else:
            return self.lm_data_loader(**kwargs)

    def bert_data_loader(self, **kwargs):
        batch = kwargs['batch']
        ctx = kwargs['ctx']
        batch_axis = kwargs['batch_axis'] or 0
        data = batch
        token_ids = data[0]
        valid_length = data[1]
        segment_ids = data[2]
        label = data[3]
        batch_size = label.shape[0]

        token_ids = gluon.utils.split_and_load(token_ids, ctx_list=ctx, batch_axis=batch_axis, even_split=False)
        valid_length = gluon.utils.split_and_load(valid_length, ctx_list=ctx, batch_axis=batch_axis,
                                                  even_split=False)
        segment_ids = gluon.utils.split_and_load(segment_ids, ctx_list=ctx, batch_axis=batch_axis, even_split=False)
        label = gluon.utils.split_and_load(label, ctx_list=ctx, batch_axis=batch_axis, even_split=False)

        ret_data = []
        for t_id, s_id, v_len in zip(token_ids, segment_ids, valid_length):
            ret_data.append((t_id, s_id, v_len.astype(np.float32)))

        return ret_data, label, batch_size

    def lm_data_loader(self, **kwargs):
        batch = kwargs['batch']
        ctx = kwargs['ctx']
        batch_axis = kwargs['batch_axis'] or 0
        data = batch[0][0]
        batch_size = data.shape[0]
        lengths = batch[0][1]
        label = batch[1]
        data = gluon.utils.split_and_load(data, ctx_list=ctx, batch_axis=batch_axis, even_split=False)
        lengths = gluon.utils.split_and_load(lengths, ctx_list=ctx, batch_axis=batch_axis, even_split=False)
        label = gluon.utils.split_and_load(label, ctx_list=ctx, batch_axis=batch_axis, even_split=False)
        ret_data = []
        for d, length in zip(data, lengths):
            ret_data.append((d.T, length.astype(np.float32)))

        return ret_data, label, batch_size
