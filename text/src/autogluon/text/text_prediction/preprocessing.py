import multiprocessing as mp
import functools
import os
from collections import OrderedDict
from typing import Dict, Optional, List, Tuple, Union
import numpy as np
from autogluon_contrib_nlp.data import batchify as bf
from autogluon_contrib_nlp.utils.preprocessing import get_trimmed_lengths, match_tokens_with_char_spans
from autogluon_contrib_nlp.utils.misc import num_mp_workers
from .dataset import TabularDataset
from .fields import TextTokenIdsField, EntityField, CategoricalField, NumericalField
from . import constants as _C

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def infer_problem_type(label_column_property):
    """Infer the type of the problem based on the column property

    Parameters
    ----------
    label_column_property

    Returns
    -------
    problem_type
        classification or regression
    problem_label_shape
        For classification problem it will be the number of classes.
        For regression problem, it will be the label shape.
    """
    if label_column_property.type == _C.CATEGORICAL:
        return _C.CLASSIFICATION, label_column_property.num_class
    elif label_column_property.type == _C.NUMERICAL:
        return _C.REGRESSION, label_column_property.shape
    else:
        raise NotImplementedError


def _chunk_processor(chunk, processing_fn):
    out = []
    for idx, row in chunk.iterrows():
        out.append(processing_fn(row))
    return out


def parallel_transform(df, processing_fn,
                       num_process=None,
                       fallback_threshold=1000):
    """Apply the function to each row of the pandas dataframe and store the results
    in a python list.

    Parameters
    ----------
    df
        Pandas Dataframe
    processing_fn
        The processing function
    num_process
        If not set. We use the default value
    fallback_threshold
        If the number of samples in df is smaller than fallback_threshold.
        Directly transform the data without multiprocessing

    Returns
    -------
    out
        List of samples
    """
    if num_process is None:
        num_process = num_mp_workers()
    if len(df) <= fallback_threshold:
        out = []
        for idx, row in df.iterrows():
            out.append(processing_fn(row))
        return out
    else:
        chunks = np.array_split(df, num_process * 8)
        with mp.Pool(num_process) as pool:
            out_l = pool.map(functools.partial(_chunk_processor, processing_fn=processing_fn),
                             chunks)
        out = sum(out_l, [])
    return out


def process_text_entity_features(
        data,
        tokenizer,
        text_columns: Optional[List[str]],
        entity_columns: Optional[List[str]],
        max_length: int,
        entity_props,
        merge_text: bool = False,
        store_token_offsets: bool = True,
        truncate_out_of_range: bool = True):
    """Process the text and entity features

    Parameters
    ----------
    data
        The input data. A Pandas Series.
    tokenizer
        The tokenizer
    text_columns
        Name of the text columns in the input data
    entity_columns
        Name of the entity columns in the input data
    max_length
        The maximum length of the merged sequence
    entity_props
        The column properties of the entities
    merge_text
        Whether to merge th individual text columns
    store_token_offsets
        Whether to get the token offsets
    truncate_out_of_range
        Whether to truncate the out-of-the-range entities.

    Returns
    -------
    text_features
        A list of TextFields
    entity_features
        A list of EntityFields
    """
    text_features = []
    entity_features = []
    # Step 1: Get the token_ids + token_offsets of all text columns.
    sentence_start_in_merged = dict()  # Store the start + end of each sentence
    sentence_slice_stat = dict()  # Store the sliced start + end of the sentence
    text_token_ids = OrderedDict()
    text_token_offsets = OrderedDict()
    for col_name in text_columns:
        token_ids, token_offsets = tokenizer.encode_with_offsets(data[col_name], int)
        token_ids = np.array(token_ids)
        token_offsets = np.array(token_offsets)
        text_token_ids[col_name] = token_ids
        text_token_offsets[col_name] = token_offsets
    lengths = [len(text_token_ids[col_name]) for col_name in text_columns]
    if merge_text:
        # We will merge the text tokens by
        # Token IDs =      [CLS]    token_ids1       [SEP]      token_ids2         [SEP]
        # Segment IDs =      0         0               0           1                 1
        # Token Offsets = (-1, -1), token_offsets1   (-1, -1)    token_offsets2   (-1, -1)
        trimmed_lengths = get_trimmed_lengths(lengths,
                                              max_length=max_length - len(lengths) - 1,
                                              do_merge=True)
        encoded_token_ids = [np.array([tokenizer.vocab.cls_id])]
        encoded_token_offsets = [np.array([[-1, -1]])]
        segment_ids = [np.array([0])]
        shift = 1
        for idx, (trim_length, col_name) in enumerate(zip(trimmed_lengths, text_columns)):
            slice_length = min(len(text_token_ids[col_name]), trim_length)
            sentence_start_in_merged[col_name] = shift
            sentence_slice_stat[col_name] = (0, slice_length)
            if slice_length > 0:
                encoded_token_ids.append(text_token_ids[col_name][:slice_length])
                segment_ids.append(np.full((slice_length,), idx % 2))
            encoded_token_ids.append(np.array([tokenizer.vocab.sep_id]))
            segment_ids.append(np.array([idx % 2]))
            if slice_length > 0:
                encoded_token_offsets.append(text_token_offsets[col_name][:slice_length])
            encoded_token_offsets.append(np.array([[-1, -1]]))
            shift += slice_length + 1
        encoded_token_ids = np.concatenate(encoded_token_ids).astype(np.int32)
        segment_ids = np.concatenate(segment_ids).astype(np.int32)
        if store_token_offsets:
            encoded_token_offsets = np.concatenate(encoded_token_offsets).astype(np.int32)
            text_features.append(TextTokenIdsField(encoded_token_ids, segment_ids,
                                                   encoded_token_offsets))
        else:
            text_features.append(TextTokenIdsField(encoded_token_ids, segment_ids,
                                                   None))
    else:
        # We encode each sentence independently
        # [CLS] token_ids1 [SEP], [CLS] token_ids2 [SEP]
        #  0     0           0  ,  0     0           0
        trimmed_lengths = get_trimmed_lengths(lengths,
                                              max_length=max_length - 2,
                                              do_merge=False)
        for trim_length, col_name in zip(trimmed_lengths, text_columns):
            slice_length = min(len(text_token_ids[col_name]), trim_length)
            sentence_slice_stat[col_name] = (0, slice_length)
            encoded_token_ids = np.concatenate([np.array([tokenizer.vocab.cls_id]),
                                                text_token_ids[col_name][:trim_length],
                                                np.array([tokenizer.vocab.sep_id])], axis=0)
            encoded_token_offsets = np.concatenate([np.array([[-1, -1]]),
                                                    text_token_offsets[col_name][:trim_length],
                                                    np.array([[-1, -1]])], axis=0)
            if store_token_offsets:
                text_features.append(TextTokenIdsField(encoded_token_ids.astype(np.int32),
                                                       np.zeros_like(encoded_token_ids,
                                                                     dtype=np.int32),
                                                       encoded_token_offsets))
            else:
                text_features.append(TextTokenIdsField(encoded_token_ids.astype(np.int32),
                                                       np.zeros_like(encoded_token_ids,
                                                                     dtype=np.int32),
                                                       None))
    # Step 2: Transform all entity columns
    for col_name in entity_columns:
        entities = data[col_name]
        col_prop = entity_props[col_name]
        parent_name = col_prop.parent
        char_offsets, transformed_labels = col_prop.transform(entities)
        # Get the stored offsets
        token_offsets = text_token_offsets[parent_name]
        entity_token_offsets = match_tokens_with_char_spans(token_offsets=token_offsets,
                                                            spans=char_offsets)
        slice_start, slice_end = sentence_slice_stat[parent_name]
        if truncate_out_of_range:
            # Ignore out-of-the-range entities
            in_bound = (entity_token_offsets[:, 0] >= slice_start) *\
                       (entity_token_offsets[:, 0] < slice_end) *\
                       (entity_token_offsets[:, 1] >= slice_start) *\
                       (entity_token_offsets[:, 1] < slice_end)
            entity_token_offsets = entity_token_offsets[in_bound]
            if transformed_labels is not None:
                transformed_labels = transformed_labels[in_bound]
        if merge_text:
            entity_token_offsets += slice_start + sentence_start_in_merged[parent_name]
        else:
            entity_token_offsets += slice_start + 1  # Add the offset w.r.t the cls token.
        entity_features.append(EntityField(entity_token_offsets, transformed_labels))
    return text_features, entity_features


class TabularBasicBERTPreprocessor:
    def __init__(self, *,
                 tokenizer,
                 column_properties,
                 max_length: int,
                 label_columns,
                 feature_columns: Optional[Union[str, List[str]]] = None,
                 store_token_offsets: bool = True,
                 merge_text: bool = True):
        """Preprocess the inputs to work with a pretrained model.

        Parameters
        ----------
        tokenizer
            The tokenizer
        column_properties
            A dictionary that contains the column properties
        max_length
            The maximum length of the encoded token sequence.
        label_columns
            The name of the label column
        feature_columns
            Names of the feature columns.
        store_token_offsets
            Whether to store the token offsets
        merge_text
            Whether to merge the token_ids when there are multiple text fields.
            For example, we will merge the text fields as
            [CLS] token_ids1 [SEP] token_ids2 [SEP] token_ids3 [SEP] token_ids4 [SEP] ...
        """
        self._tokenizer = tokenizer
        self._column_properties = column_properties
        if isinstance(label_columns, str):
            self._label_columns = [label_columns]
        else:
            self._label_columns = label_columns
        assert len(self._label_columns) > 0, 'Must specify the label_columns!'
        for col_name in self._label_columns:
            assert col_name in column_properties, 'label_column="{}" is not found ' \
                                                  'in column property'.format(col_name)
        if feature_columns is not None:
            self._feature_columns = feature_columns
        else:
            self._feature_columns = [key
                                     for key in self._column_properties.keys() if key not in self._label_columns]
        self._max_length = max_length
        self._merge_text = merge_text
        self._store_token_offsets = store_token_offsets
        self._text_columns = []
        self._entity_columns = []
        self._categorical_columns = []
        self._numerical_columns = []
        for col_name, col_info in self._column_properties.items():
            if col_name in self.label_columns:
                assert col_info.type == _C.CATEGORICAL or col_info.type == _C.NUMERICAL
            if col_info.type == _C.TEXT:
                self._text_columns.append(col_name)
            elif col_info.type == _C.ENTITY:
                self._entity_columns.append(col_name)
            elif col_info.type == _C.CATEGORICAL:
                self._categorical_columns.append(col_name)
            elif col_info.type == _C.NUMERICAL:
                self._numerical_columns.append(col_name)
            else:
                raise NotImplementedError
        self._text_column_require_offsets = {col_name: False for col_name in self.text_columns}
        for col_name in self._entity_columns:
            self._text_column_require_offsets[self.column_properties[col_name].parent] = True

    @property
    def feature_columns(self):
        return self._feature_columns

    @property
    def label_columns(self):
        return self._label_columns

    @property
    def max_length(self):
        return self._max_length

    @property
    def column_properties(self):
        return self._column_properties

    @property
    def merge_text(self):
        return self._merge_text

    @property
    def text_columns(self):
        return self._text_columns

    @property
    def text_column_require_offsets(self):
        return self._text_column_require_offsets

    @property
    def entity_columns(self):
        return self._entity_columns

    @property
    def categorical_columns(self):
        return self._categorical_columns

    @property
    def numerical_columns(self):
        return self._numerical_columns

    def feature_field_info(self):
        """Get the field information of the features after this transformation

        Returns
        -------
        info_l
            A list that stores the status of the output features.
        """
        info_l = []
        text_col_idx = dict()
        if len(self.text_columns) > 0:
            if self.merge_text:
                info_l.append((_C.TEXT, dict()))
            else:
                for i, col_name in enumerate(self.text_columns):
                    text_col_idx[col_name] = i
                    info_l.append((_C.TEXT, dict()))
        if len(self.entity_columns) > 0:
            for col_name in self.entity_columns:
                if col_name in self.label_columns:
                    continue
                parent = self.column_properties[col_name].parent
                if self.merge_text:
                    parent_idx = 0
                else:
                    parent_idx = text_col_idx[parent]
                info_l.append((_C.ENTITY,
                               {'parent_idx': parent_idx,
                                'prop': self.column_properties[col_name]}))
        if len(self.categorical_columns) > 0:
            for col_name in self.categorical_columns:
                if col_name in self.label_columns:
                    continue
                info_l.append((_C.CATEGORICAL,
                               {'prop': self.column_properties[col_name]}))
        if len(self.numerical_columns) > 0:
            for col_name in self.numerical_columns:
                if col_name in self.label_columns:
                    continue
                info_l.append((_C.NUMERICAL,
                               {'prop': self.column_properties[col_name]}))
        return info_l

    def label_field_info(self):
        """

        Returns
        -------
        info_l
            A list of label info
        """
        info_l = []
        for col_name in self.label_columns:
            col_prop = self.column_properties[col_name]
            if col_prop.type == _C.CATEGORICAL:
                info_l.append((_C.CATEGORICAL, {'prop': self.column_properties[col_name]}))
            elif col_prop.type == _C.NUMERICAL:
                info_l.append((_C.NUMERICAL, {'prop': self.column_properties[col_name]}))
            else:
                raise NotImplementedError
        return info_l

    def batchify(self, round_to=None, is_test=False):
        """

        Parameters
        ----------
        round_to
            Whether to round to a specific multiplier when calling PadBatchify
        is_test
            Whether the batchify function is for training

        Returns
        -------
        batchify_fn
            The batchify function
        """
        feature_batchify_fn_l = []
        for type_code, attrs in self.feature_field_info():
            if type_code == _C.TEXT:
                feature_batchify_fn_l.append(TextTokenIdsField.batchify(round_to))
            elif type_code == _C.ENTITY:
                feature_batchify_fn_l.append(EntityField.batchify())
            elif type_code == _C.CATEGORICAL:
                feature_batchify_fn_l.append(CategoricalField.batchify())
            elif type_code == _C.NUMERICAL:
                feature_batchify_fn_l.append(NumericalField.batchify())
            else:
                raise NotImplementedError
        if is_test:
            return bf.Tuple(feature_batchify_fn_l)
        else:
            label_batchify_fn_l = []
            for type_code, attrs in self.label_field_info():
                if type_code == _C.CATEGORICAL:
                    label_batchify_fn_l.append(CategoricalField.batchify())
                elif type_code == _C.NUMERICAL:
                    label_batchify_fn_l.append(NumericalField.batchify())
                else:
                    raise NotImplementedError
            return bf.Tuple(bf.Tuple(feature_batchify_fn_l),
                            bf.Tuple(label_batchify_fn_l))

    def process_train(self, df_or_dataset):
        if isinstance(df_or_dataset, TabularDataset):
            df_or_dataset = df_or_dataset.table
        return parallel_transform(df_or_dataset, functools.partial(self.__call__, is_test=False))

    def process_test(self, df_or_dataset):
        if isinstance(df_or_dataset, TabularDataset):
            df_or_dataset = df_or_dataset.table
        return parallel_transform(df_or_dataset, functools.partial(self.__call__, is_test=True))

    def __call__(self, data, is_test=False):
        """Transform the data into a list of fields.

        Here, the sample can either be a row in pandas dataframe or a named-tuple.

        We organize and represent the features in the following format:

        - Text fields
            We transform text into a sequence of token_ids.
            If there are multiple text fields, we have the following options
            1) merge_text = True
                We will concatenate these text fields and inserting CLS, SEP ids, i.e.
                [CLS] text_ids1 [SEP] text_ids2 [SEP]
            2) merge_text = False
                We will transform each text field separately:
                [CLS] text_ids1 [SEP], [CLS] text_ids2 [SEP], ...
            For empty text / missing text data, we will just convert it to [CLS] [SEP]
        - Entity fields
            The raw entities are stored as character-level start and end offsets.
            After the preprocessing, we will store them as the token-level
            start + end. Different from the raw character-level start + end offsets, the
            token-level start + end offsets will be used.
            - token_level_start, token_level_end, span_label
            or
            - token_level_start, token_level_end
        - Categorical fields
            We transform the categorical features to its ids.
        - Numerical fields
            We keep the numerical features and indicate the missing value

        Parameters
        ----------
        data
            A single data sample.

        Returns
        -------
        features
            Preprocessed features. Will contain the following
            - TEXT
                The encoded value will be a TextTokenIdsField

            - ENTITY
                The encoded feature will be:
                data: Shape (num_entity, 2)
                    Each item will be (start, end)
                if has_label:
                    label:
                        - Categorical: Shape (num_entity,)
                        - Numerical: (num_entity,) + label_shape

            - CATEGORICAL
                The categorical feature. Will be an integer

            - NUMERICAL
                The numerical feature. Will be a numpy array
        labels
            The preprocessed labels
        """
        feature_fields = []
        # Step 1: Get the text features + entity features
        text_fields, entity_fields =\
            process_text_entity_features(
                data=data,
                tokenizer=self._tokenizer,
                text_columns=self.text_columns,
                entity_columns=self.entity_columns,
                max_length=self.max_length,
                entity_props=OrderedDict([(col_name, self.column_properties[col_name])
                                          for col_name in self.entity_columns]),
                merge_text=self.merge_text,
                store_token_offsets=self._store_token_offsets,
                truncate_out_of_range=True)
        feature_fields.extend(text_fields)
        feature_fields.extend(entity_fields)

        # Step 2: Transform all categorical columns
        categorical_fields = []
        for col_name in self.categorical_columns:
            if col_name in self.label_columns:
                continue
            col_prop = self.column_properties[col_name]
            transformed_labels = col_prop.transform(data[col_name])
            categorical_fields.append(CategoricalField(transformed_labels))
        feature_fields.extend(categorical_fields)

        # Step 4: Transform all numerical columns
        numerical_fields = []
        for col_name in self.numerical_columns:
            if col_name in self.label_columns:
                continue
            col_prop = self.column_properties[col_name]
            numerical_fields.append(NumericalField(col_prop.transform(data[col_name])))
        feature_fields.extend(numerical_fields)
        if is_test:
            return tuple(feature_fields)
        else:
            label_fields = []
            for col_name in self.label_columns:
                col_prop = self.column_properties[col_name]
                if col_prop.type == _C.CATEGORICAL:
                    label_fields.append(CategoricalField(col_prop.transform(data[col_name])))
                elif col_prop.type == _C.NUMERICAL:
                    label_fields.append(NumericalField(col_prop.transform(data[col_name])))
                else:
                    raise NotImplementedError
            return tuple(feature_fields), tuple(label_fields)
