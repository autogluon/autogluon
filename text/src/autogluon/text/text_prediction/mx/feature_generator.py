from autogluon_contrib_nlp.utils.config import CfgNode
from autogluon_contrib_nlp.utils.preprocessing import get_trimmed_lengths
from sklearn.base import TransformerMixin, BaseEstimator
from .. import constants as _C


def base_preprocess_cfg():
    cfg = CfgNode()
    cfg.text = CfgNode()
    cfg.text.merge = True                     # Whether we will merge different text columns
                                              # or treat them independently.
    cfg.text.max_length = 512                 # The maximum possible length.
    cfg.text.auto_max_length = True           # Try to automatically shrink the maximal length
                                              # based on the statistics of the dataset.
    cfg.categorical = CfgNode()
    cfg.categorical.minimum_cat_count = 100   # The minimal number of data per categorical group
    cfg.categorical.maximum_num_cat = 20      # The minimal number of data per categorical group
    cfg.categorical.convert_to_text = False   # Whether to convert the feature to text

    cfg.numerical = CfgNode()
    cfg.numerical.convert_to_text = False     # Whether to convert the feature to text
    cfg.numerical.impute_strategy = 'mean'    # Whether to use mean to fill in the missing values.
    return cfg


def tokenizer_columns():
    


class MultiModalTextModelDataTransformer(TransformerMixin, BaseEstimator):
    def __init__(self, column_types, label_column, tokenizer, cfg=None):
        self._column_types = column_types
        self._label_column = label_column
        self._cfg = cfg
        self._generators = dict()
        for col_name, col_type in self._column_types:
            if col_type == _C.TEXT:
                continue
            elif col_type == _C.CATEGORICAL:
                continue
        self._tokenizer = tokenizer

    def fit_transform(self, data_df):
        """

        Parameters
        ----------
        data_df

        Returns
        -------

        """

    def fit(self, data_df):
        """

        Parameters
        ----------
        data_df

        Returns
        -------

        """

    def transform(self, data_df):
        """"

        """

