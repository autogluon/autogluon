from . import constants as _C
from autogluon_contrib_nlp.data import batchify as bf


class _TextTokenIdsFieldBatchify:
    def __init__(self, round_to=None):
        self._pad_batchify = bf.Pad(round_to=round_to)
        self._stack_batchify = bf.Stack()

    def __call__(self, data):
        """

        Parameters
        ----------
        data

        Returns
        -------
        batch_token_ids
            (batch_size, sequence_length)
        batch_valid_length
            (batch_size,)
        batch_segment_ids
            (batch_size, sequence_length)
        batch_token_offsets
            (batch_size, sequence_length, 2)
        """
        batch_token_ids = self._pad_batchify([ele.token_ids for ele in data])
        batch_valid_length = self._stack_batchify([len(ele.token_ids) for ele in data])
        if data[0].segment_ids is None:
            batch_segment_ids = None
        else:
            batch_segment_ids = self._pad_batchify([ele.segment_ids for ele in data])
        if data[0].token_offsets is None:
            batch_token_offsets = None
        else:
            batch_token_offsets = self._pad_batchify([ele.token_offsets for ele in data])
        return batch_token_ids, batch_valid_length, batch_segment_ids, batch_token_offsets


class TextTokenIdsField:
    type = _C.TEXT

    def __init__(self, token_ids, segment_ids=None, token_offsets=None):
        """

        Parameters
        ----------
        token_ids
            The token_ids, shape (seq_length,)
        segment_ids
            The segment_ids, shape (seq_length,)
        token_offsets
            The character-level offsets of the token, shape (seq_length, 2)
        """
        self.token_ids = token_ids
        self.segment_ids = segment_ids
        self.token_offsets = token_offsets

    @staticmethod
    def batchify(round_to=None):
        return _TextTokenIdsFieldBatchify(round_to)

    def __str__(self):
        ret = '{}(\n'.format(self.__class__.__name__)
        ret += 'token_ids={}\n'.format(self.token_ids)
        ret += 'segment_ids={}\n'.format(self.segment_ids)
        ret += 'token_offsets={}\n'.format(self.token_offsets)
        ret += ')\n'
        return ret


class _EntityFieldBatchify:
    def __init__(self):
        self._pad_batchify = bf.Pad()
        self._stack_batchify = bf.Stack()

    def __call__(self, data):
        """The internal batchify function

        Parameters
        ----------
        data
            The input data.

        Returns
        -------
        batch_span
            Shape (batch_size, #num_entities, 2)
        batch_label
            Shape (batch_size, #num_entities) + label_shape
        batch_num_entity
            Shape (batch_size,)
        """
        batch_span = self._pad_batchify([ele.data for ele in data])
        no_label = data[0].label is None
        if no_label:
            batch_label = None
        else:
            batch_label = self._pad_batchify([ele.label for ele in data])
        batch_num_entity = self._stack_batchify([len(ele.data) for ele in data])
        return batch_span, batch_label, batch_num_entity


class EntityField:
    type = _C.ENTITY

    def __init__(self, data, label=None):
        """

        Parameters
        ----------
        data
            (#Num Entities, 2)
        label
            (#Num Entities,)
        """
        self.data = data
        self.label = label

    @staticmethod
    def batchify():
        return _EntityFieldBatchify()

    def __str__(self):
        ret = '{}(\n'.format(self.__class__.__name__)
        ret += 'data={}\n'.format(self.data)
        ret += 'label={}\n'.format(None if self.label is None else self.label)
        ret += ')\n'
        return ret


class _ArrayBatchify:
    def __init__(self):
        self._stack_batchify = bf.Stack()

    def __call__(self, data):
        """

        Parameters
        ----------
        data

        Returns
        -------
        dat
            Shape (batch_size,) + sample_shape
        """
        return self._stack_batchify([ele.data for ele in data])


class NumericalField:
    type = _C.NUMERICAL

    def __init__(self, data):
        self.data = data

    @staticmethod
    def batchify():
        return _ArrayBatchify()

    def __str__(self):
        ret = '{}(\n'.format(self.__class__.__name__)
        ret += 'data={}\n'.format(self.data)
        ret += ')\n'
        return ret


class CategoricalField(NumericalField):
    type = _C.CATEGORICAL
    pass
