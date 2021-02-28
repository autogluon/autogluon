import numpy as np
import os
import pandas as pd
import functools
import logging
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from mxnet.gluon.data import ArrayDataset
from autogluon_contrib_nlp.utils.config import CfgNode
from autogluon_contrib_nlp.models import get_backbone
from autogluon_contrib_nlp.data.batchify import Pad, Stack, Tuple
from autogluon.features import CategoryFeatureGenerator

from .. import constants as _C
from ..utils import parallel_transform, get_trimmed_lengths

os.environ["TOKENIZERS_PARALLELISM"] = "false"

logger = logging.getLogger(__name__)


def base_preprocess_cfg():
    cfg = CfgNode()
    cfg.text = CfgNode()
    cfg.text.merge = True                     # Whether we will merge different text columns
                                              # or treat them independently.
    cfg.text.max_length = 512                 # The maximum possible length.
    cfg.text.auto_max_length = True           # Try to automatically shrink the maximal length
                                              # based on the statistics of the dataset.
    cfg.text.auto_max_length_quantile = 0.95  # We will ensure that the new max_length is around the quantile of the lengths of all samples
    cfg.text.auto_max_length_round_to = 32    # We will ensure that the automatically determined max length will be divisible by round_to
    cfg.categorical = CfgNode()
    cfg.categorical.minimum_cat_count = 100   # The minimal number of data per categorical group
    cfg.categorical.maximum_num_cat = 20      # The minimal number of data per categorical group
    cfg.categorical.convert_to_text = False   # Whether to convert the feature to text

    cfg.numerical = CfgNode()
    cfg.numerical.convert_to_text = False     # Whether to convert the feature to text
    cfg.numerical.impute_strategy = 'mean'    # Whether to use mean to fill in the missing values.
                                              # We use the imputer in sklearn: https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html
                                              # The strategies can thus be "mean", "median", "most_frequent".
    cfg.numerical.scaler_with_mean = True     # Whether to normalize with mean
    cfg.numerical.scaler_with_std = True      # Whether to normalize with std
    cfg.freeze()
    return cfg


def tokenize_data(data: pd.Series, tokenizer):
    out = []
    if data is not None:
        for idx, ele in data.iteritems():
            if ele is None:
                out.append(np.ones((0,), dtype=np.int32))
            else:
                out.append(np.array(tokenizer.encode(ele, int), dtype=np.int32))
    return out


def get_tokenizer(backbone_name):
    _, _, tokenizer, _, _ = get_backbone(backbone_name)
    return tokenizer


class MultiModalTextBatchify:
    def __init__(self,
                 num_text_inputs,
                 num_categorical_inputs,
                 num_numerical_inputs,
                 cls_token_id,
                 sep_token_id,
                 max_length,
                 num_segments=2,
                 mode='train',
                 stochastic_chunk=True,
                 insert_sep=True):
        """Batchify function for the multimodal text column

        Parameters
        ----------
        num_text_inputs
            number of text inputs
        num_categorical_inputs
            number of categorical inputs
        num_numerical_inputs
            number of numerical inputs
        cls_token_id
            CLS ID
        sep_token_id
        max_length
        num_segments
            The supported number of segments of the pretraiend backbone.
            For BERT + ELECTRA, it's 2. For RoBERTa, it's 1
        mode
        stochastic_chunk
        insert_sep
            Whether to insert sep
        """
        self._insert_sep = insert_sep
        self._mode = mode
        self._cls_token_id = cls_token_id
        self._sep_token_id = sep_token_id
        self._stochastic_chunk = stochastic_chunk
        self._num_text_inputs = num_text_inputs
        self._num_categorical_inputs = num_categorical_inputs
        self._num_numerical_inputs = num_numerical_inputs
        self._max_length = max_length
        self._num_segments = num_segments
        self._pad_batchify = Pad()
        self._stack_batchify = Stack()
        if self._num_categorical_inputs > 0:
            self._categorical_batchify = Tuple([Stack()
                                                for _ in range(self._num_categorical_inputs)])
        else:
            self._categorical_batchify = None
        assert self._num_numerical_inputs == 0 or self._num_numerical_inputs == 1

    @property
    def num_input_features(self):
        return self._num_text_inputs + self._num_categorical_inputs + self._num_numerical_inputs

    @property
    def num_text_outputs(self):
        return 1

    @property
    def num_categorical_outputs(self):
        return self._num_categorical_inputs

    @property
    def num_numerical_outputs(self):
        return self._num_numerical_inputs

    def __call__(self, samples):
        text_token_ids = []
        text_valid_length = []
        text_segment_ids = []
        categorical_features = []
        numerical_features = []
        labels = []
        for ele in samples:
            if not isinstance(ele, tuple):
                ele = (ele,)
            # Get text features
            if self._insert_sep:
                max_length = self._max_length - (self._num_text_inputs + 1)
            else:
                max_length = self._max_length - 2
            trimmed_lengths = get_trimmed_lengths([len(ele[i])
                                                   for i in range(self._num_text_inputs)],
                                                  max_length,
                                                  do_merge=True)
            seg = 0
            token_ids = [self._cls_token_id]
            segment_ids = [seg]
            for i, trim_length in enumerate(trimmed_lengths):
                if self._stochastic_chunk:
                    start_ptr = np.random.randint(0, len(ele[i]) - trim_length + 1)
                else:
                    start_ptr = 0
                token_ids.extend(ele[i][start_ptr:(start_ptr + trim_length)].tolist())
                segment_ids.extend([seg] * trim_length)
                if self._insert_sep or i == len(trimmed_lengths) - 1:
                    token_ids.append(self._sep_token_id)
                    segment_ids.append(seg)
                seg = (seg + 1) % self._num_segments
            text_token_ids.append(np.array(token_ids, dtype=np.int32))
            text_valid_length.append(len(token_ids))
            text_segment_ids.append(np.array(segment_ids, dtype=np.int32))
            # Get categorical features
            ptr = self._num_text_inputs
            if self._num_categorical_inputs > 0:
                categorical_features.append(ele[ptr:(ptr + self._num_categorical_inputs)])
            ptr += self._num_categorical_inputs

            # Get numerical features
            if self._num_numerical_inputs > 0:
                numerical_features.append(ele[ptr].astype(np.float32))
            ptr += self._num_numerical_inputs
            if self._mode == 'train':
                labels.append(ele[ptr])
        features = []
        features.append((self._pad_batchify(text_token_ids),
                         self._stack_batchify(text_valid_length),
                         self._pad_batchify(text_segment_ids)))
        if self._num_categorical_inputs > 0:
            features.extend(self._categorical_batchify(categorical_features))
        if self._num_numerical_inputs > 0:
            features.append(self._stack_batchify(numerical_features))
        if self._mode == 'train':
            labels = self._stack_batchify(labels)
            return features, labels
        else:
            return features


def auto_shrink_max_length(train_dataset, insert_sep,
                           num_text_features,
                           auto_max_length_quantile,
                           round_to,
                           max_length):
    """Automatically shrink the max length based on the training data

    Parameters
    ----------
    train_dataset
        The training dataset
    insert_sep
    num_text_features
    auto_max_length_quantile
    round_to
    max_length

    Returns
    -------
    new_max_length
    """
    lengths = []
    for sample in train_dataset:
        if insert_sep:
            lengths.append(num_text_features + 1 + sum([len(sample[i])
                                                        for i in range(num_text_features)]))
        else:
            lengths.append(2 + sum([len(sample[i]) for i in range(num_text_features)]))
    real_data_max_length = max(lengths)
    if real_data_max_length >= max_length:
        quantile_length = np.quantile(lengths, auto_max_length_quantile)
        quantile_length = int(round_to * np.ceil(quantile_length / round_to))
        return min(quantile_length, max_length)
    else:
        return min(int(round_to * np.ceil(real_data_max_length / round_to)), max_length)


def get_stats_string(processor, dataset, is_train=False):
    ret = 'Features:\n'
    ret += '   Text Column:\n'
    for i, col_name in enumerate(processor.text_feature_names):
        lengths = [len(ele[i]) for ele in dataset]
        ret += '      - "{}":' \
               ' Tokenized Length Min/Avg/Max=' \
               '{}/{:.2f}/{}\n'.format(col_name, np.min(lengths),
                                       np.mean(lengths),
                                       np.max(lengths))
    ret += '   Categorical Column:\n'
    for col_name, num_category in zip(processor.categorical_feature_names,
                                      processor.categorical_num_categories):
        ret += f'      - "{col_name}": Num Class={num_category}\n'
    ret += f'   Numerical Columns: \n'
    for col_name in processor.numerical_feature_names:
        ret += f'      - "{col_name}"\n'
    if is_train:
        ret += f'Label: "{processor.label_column}"'
        if processor._column_types[processor.label_column] == _C.CATEGORICAL:
            ret += f', Classes={processor.label_generator.classes_}\n'
    return ret


def get_cls_sep_id(tokenizer):
    if hasattr(tokenizer.vocab, 'cls_id'):
        cls_id = tokenizer.vocab.cls_id
        sep_id = tokenizer.vocab.sep_id
    elif hasattr(tokenizer.vocab, 'bos_id'):
        cls_id = tokenizer.vocab.bos_id
        sep_id = tokenizer.vocab.eos_id
    else:
        raise NotImplementedError
    return cls_id, sep_id


class MultiModalTextFeatureProcessor(TransformerMixin, BaseEstimator):
    def __init__(self, column_types, label_column, tokenizer_name, label_generator=None, cfg=None):
        self._column_types = column_types
        self._label_column = label_column
        cfg = base_preprocess_cfg().clone_merge(cfg)
        self._cfg = cfg
        self._feature_generators = dict()
        self._label_generator = label_generator
        self._label_scaler = StandardScaler()   # Scaler used for numerical labels
        for col_name, col_type in self._column_types.items():
            if col_name == self._label_column:
                continue
            if col_type == _C.TEXT:
                continue
            elif col_type == _C.CATEGORICAL:
                generator = CategoryFeatureGenerator(
                    cat_order='count',
                    minimum_cat_count=cfg.categorical.minimum_cat_count,
                    maximum_num_cat=cfg.categorical.maximum_num_cat,
                    verbosity=0)
                self._feature_generators[col_name] = generator
            elif col_type == _C.NUMERICAL:
                generator = Pipeline(
                    [('imputer', SimpleImputer()),
                     ('scaler', StandardScaler(with_mean=cfg.numerical.scaler_with_mean,
                                               with_std=cfg.numerical.scaler_with_std))]
                )
                self._feature_generators[col_name] = generator

        self._tokenizer_name = tokenizer_name
        self._tokenizer = get_tokenizer(tokenizer_name)
        self._fit_called = False

        # Some columns will be ignored
        self._ignore_columns_set = set()
        self._text_feature_names = []
        self._categorical_feature_names = []
        self._categorical_num_categories = []
        self._numerical_feature_names = []

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def label_column(self):
        return self._label_column

    @property
    def text_feature_names(self):
        return self._text_feature_names

    @property
    def categorical_feature_names(self):
        return self._categorical_feature_names

    @property
    def numerical_feature_names(self):
        return self._numerical_feature_names

    @property
    def categorical_num_categories(self):
        """We will always include the unknown category"""
        return self._categorical_num_categories

    @property
    def cfg(self):
        return self._cfg

    @property
    def label_type(self):
        return self._column_types[self._label_column]

    @property
    def label_scaler(self):
        return self._label_scaler

    @property
    def label_generator(self):
        return self._label_generator

    def fit_transform(self, X, y):
        """Fit and Transform the dataframe

        Parameters
        ----------
        X
            The feature dataframe
        y
            The label series

        Returns
        -------
        processed_X
            The processed X data
        (processed_y)
            The processed Y data
        """
        if self._fit_called:
            raise RuntimeError('Fit has been called. Please create a new preprocessor and call '
                               'fit again!')
        self._fit_called = True
        text_features = []
        categorical_features = []
        numerical_features = []
        for col_name in sorted(X.columns):
            col_type = self._column_types[col_name]
            logger.log(10, f'Process col "{col_name}" with type "{col_type}"')
            col_value = X[col_name]
            if col_type == _C.NULL:
                self._ignore_columns_set.add(col_name)
                continue
            elif col_type == _C.TEXT:
                col_value = col_value.apply(lambda ele: '' if ele is None else str(ele))
                processed_col_value = parallel_transform(
                    df=col_value,
                    chunk_processor=functools.partial(tokenize_data,
                                                      tokenizer=self._tokenizer))
                text_features.append(processed_col_value)
                self._text_feature_names.append(col_name)
            elif col_type == _C.CATEGORICAL:
                if self.cfg.categorical.convert_to_text:
                    # Convert categorical column as text column
                    processed_data = col_value.apply(lambda ele: '' if ele is None else str(ele))
                    if len(np.unique(processed_data)) == 1:
                        self._ignore_columns_set.add(col_name)
                        continue
                    processed_data = parallel_transform(
                        df=processed_data,
                        chunk_processor=functools.partial(tokenize_data, tokenizer=self._tokenizer))
                    text_features.append(processed_data)
                    self._text_feature_names.append(col_name)
                else:
                    processed_data = col_value.astype('category')
                    generator = self._feature_generators[col_name]
                    processed_data = generator.fit_transform(
                        pd.DataFrame({col_name: processed_data}))[col_name]\
                        .cat.codes.to_numpy(np.int32, copy=True)
                    if len(np.unique(processed_data)) == 1:
                        self._ignore_columns_set.add(col_name)
                        continue
                    num_categories = len(generator.category_map[col_name])
                    processed_data[processed_data < 0] = num_categories
                    self._categorical_num_categories.append(num_categories + 1)
                    categorical_features.append(processed_data)
                    self._categorical_feature_names.append(col_name)
            elif col_type == _C.NUMERICAL:
                processed_data = pd.to_numeric(col_value)
                if len(processed_data.unique()) == 1:
                    self._ignore_columns_set.add(col_name)
                    continue
                if self.cfg.numerical.convert_to_text:
                    processed_data = processed_data.apply('{:.3f}'.format)
                    processed_data = parallel_transform(
                        df=processed_data,
                        chunk_processor=functools.partial(tokenize_data, tokenizer=self._tokenizer))
                    text_features.append(processed_data)
                    self._text_feature_names.append(col_name)
                else:
                    generator = self._feature_generators[col_name]
                    processed_data = generator.fit_transform(
                        np.expand_dims(processed_data.to_numpy(), axis=-1))[:, 0]
                    numerical_features.append(processed_data.astype(np.float32))
                    self._numerical_feature_names.append(col_name)
            else:
                raise NotImplementedError(f'Type of the column is not supported currently. '
                                          f'Received {col_name}={col_type}.')
        if len(numerical_features) > 0:
            numerical_features = [np.stack(numerical_features, axis=-1)]
        if self.label_type == _C.CATEGORICAL:
            if self._label_generator is None:
                self._label_generator = LabelEncoder()
                y = self._label_generator.fit_transform(y)
            else:
                y = self._label_generator.transform(y)
        elif self.label_type == _C.NUMERICAL:
            y = pd.to_numeric(y).to_numpy()
            y = self._label_scaler.fit_transform(np.expand_dims(y, axis=-1))[:, 0].astype(np.float32)
        else:
            raise NotImplementedError(f'Type of label column is not supported. '
                                      f'Label column type={self._label_column}')
        # Wrap the processed features and labels into a training dataset
        all_data = text_features + categorical_features + numerical_features + [y]
        dataset = ArrayDataset(*all_data)
        return dataset

    def transform(self, X_df, y_df=None):
        """"Transform the columns"""
        assert self._fit_called, 'You will need to first call ' \
                                 'preprocessor.fit_transform before calling ' \
                                 'preprocessor.transform.'
        text_features = []
        categorical_features = []
        numerical_features = []
        for col_name in self._text_feature_names:
            col_value = X_df[col_name]
            col_type = self._column_types[col_name]
            if col_type == _C.TEXT or col_type == _C.CATEGORICAL:
                processed_data = col_value.apply(lambda ele: '' if ele is None else str(ele))
            elif col_type == _C.NUMERICAL:
                processed_data = pd.to_numeric(col_value).apply('{:.3f}'.format)
            else:
                raise  NotImplementedError
            processed_data = parallel_transform(
                df=processed_data,
                chunk_processor=functools.partial(tokenize_data,
                                                  tokenizer=self._tokenizer))
            text_features.append(processed_data)

        for col_name, num_category in zip(self._categorical_feature_names,
                                          self._categorical_num_categories):
            col_value = X_df[col_name]
            processed_data = col_value.astype('category')
            generator = self._feature_generators[col_name]
            processed_data = generator.transform(
                pd.DataFrame({col_name: processed_data}))[col_name] \
                .cat.codes.to_numpy(np.int32, copy=True)
            processed_data[processed_data < 0] = num_category - 1
            categorical_features.append(processed_data)

        for col_name in self._numerical_feature_names:
            generator = self._feature_generators[col_name]
            col_value = pd.to_numeric(X_df[col_name]).to_numpy()
            processed_data = generator.transform(np.expand_dims(col_value, axis=-1))[:, 0]
            numerical_features.append(processed_data.astype(np.float32))
        if len(numerical_features) > 0:
            numerical_features = [np.stack(numerical_features, axis=-1)]
        if y_df is not None:
            if self.label_type == _C.CATEGORICAL:
                y = self.label_generator.transform(y_df)
            elif self.label_type == _C.NUMERICAL:
                y = pd.to_numeric(y_df).to_numpy()
                y = self.label_scaler.transform(np.expand_dims(y, axis=-1))[:, 0].astype(np.float32)
            else:
                raise NotImplementedError
            all_data = text_features + categorical_features + numerical_features + [y]
            return ArrayDataset(*all_data)
        else:
            all_data = text_features + categorical_features + numerical_features
            return ArrayDataset(*all_data)

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_cfg'] = self._cfg.dump()
        state['_tokenizer'] = None
        state['_logger'] = None
        return state

    def __setstate__(self, state):
        tokenizer_name = state['_tokenizer_name']
        tokenizer = get_tokenizer(tokenizer_name)
        state['_tokenizer'] = tokenizer
        state['_logger'] = logging
        state['_cfg'] = CfgNode.load_cfg(state['_cfg'])
        self.__dict__ = state
