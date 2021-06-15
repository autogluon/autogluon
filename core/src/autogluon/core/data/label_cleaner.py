import copy
import logging
from typing import Union

import numpy as np
from pandas import DataFrame, Series

from autogluon.core.constants import BINARY, MULTICLASS, REGRESSION, QUANTILE, SOFTCLASS

from autogluon.core.utils import get_pred_from_proba, get_pred_from_proba_df

logger = logging.getLogger(__name__)


# LabelCleaner cleans labels prior to entering feature generation
class LabelCleaner:
    num_classes = None
    inv_map = None
    ordered_class_labels = None
    ordered_class_labels_transformed = None
    problem_type_transform = None

    @staticmethod
    def construct(problem_type: str, y: Union[Series, np.ndarray, list, DataFrame], y_uncleaned: Union[Series, np.ndarray, list, DataFrame] = None, positive_class=None):
        if problem_type == SOFTCLASS:
            return LabelCleanerSoftclass(y)
        y = LabelCleaner._convert_to_valid_series(y)
        if y_uncleaned is not None:
            y_uncleaned = LabelCleaner._convert_to_valid_series(y_uncleaned)

        if problem_type == BINARY:
            return LabelCleanerBinary(y, positive_class=positive_class)
        elif problem_type == MULTICLASS:
            if y_uncleaned is None:
                y_uncleaned = copy.deepcopy(y)
            if len(y.unique()) == 2:
                return LabelCleanerMulticlassToBinary(y, y_uncleaned)
            else:
                return LabelCleanerMulticlass(y, y_uncleaned)
        elif problem_type in [REGRESSION, QUANTILE]:
            return LabelCleanerDummy(problem_type=problem_type)
        else:
            raise NotImplementedError

    def transform(self, y: Union[Series, np.ndarray, list]) -> Series:
        y = self._convert_to_valid_series(y)
        return self._transform(y)

    def inverse_transform(self, y: Union[Series, np.ndarray, list]) -> Series:
        y = self._convert_to_valid_series(y)
        return self._inverse_transform(y)

    def _transform(self, y: Series) -> Series:
        raise NotImplementedError

    def _inverse_transform(self, y: Series) -> Series:
        raise NotImplementedError

    def transform_proba(self, y: Union[DataFrame, Series, np.ndarray], as_pandas=False):
        return y

    def inverse_transform_proba(self, y, as_pandas=False, as_pred=False):
        return y

    @staticmethod
    def _convert_to_valid_series(y: Union[Series, np.ndarray, list]) -> Series:
        if isinstance(y, np.ndarray) or isinstance(y, list):
            y = Series(y)
        elif isinstance(y, Series) and y.dtype.name == 'category':
            y = y.astype('object')
        return y


class LabelCleanerMulticlass(LabelCleaner):
    def __init__(self, y: Series, y_uncleaned: Series):
        self.problem_type_transform = MULTICLASS
        y = self._convert_to_valid_series(y)
        y_uncleaned = self._convert_to_valid_series(y_uncleaned)
        self.cat_mappings_dependent_var: dict = self._generate_categorical_mapping(y)
        self.inv_map: dict = {v: k for k, v in self.cat_mappings_dependent_var.items()}

        self.cat_mappings_dependent_var_uncleaned: dict = self._generate_categorical_mapping(y_uncleaned)
        self.inv_map_uncleaned: dict = {v: k for k, v in self.cat_mappings_dependent_var_uncleaned.items()}

        self.num_classes = len(self.cat_mappings_dependent_var.keys())
        self.ordered_class_labels = list(y_uncleaned.astype('category').cat.categories)
        self.valid_ordered_class_labels = list(y.astype('category').cat.categories)
        self.ordered_class_labels_transformed = list(range(len(self.valid_ordered_class_labels)))
        self.invalid_class_count = len(self.ordered_class_labels) - len(self.valid_ordered_class_labels)
        self.labels_to_zero_fill = [1 if label not in self.valid_ordered_class_labels else 0 for label in self.ordered_class_labels]
        self.label_index_to_keep = [i for i, label in enumerate(self.labels_to_zero_fill) if label == 0]
        self.label_index_to_remove = [i for i, label in enumerate(self.labels_to_zero_fill) if label == 1]

    def _transform(self, y: Series) -> Series:
        y = y.map(self.inv_map)
        return y

    def _inverse_transform(self, y: Series) -> Series:
        y = y.map(self.cat_mappings_dependent_var)
        return y

    # TODO: Unused? There are not many reasonable situations that seem to require this method.
    def transform_proba(self, y: Union[DataFrame, np.ndarray], as_pandas=False) -> Union[DataFrame, np.ndarray]:
        if self.invalid_class_count > 0:
            # this assumes y has only 0's for any columns it is about to remove, if it does not, weird things may start to happen since rows will not sum to 1
            if isinstance(y, DataFrame):
                cols_to_drop = [self.cat_mappings_dependent_var_uncleaned[col] for col in self.label_index_to_remove]
                y = y.drop(columns=cols_to_drop)
            else:
                y = np.delete(y, self.label_index_to_remove, axis=1)
        if as_pandas:
            if isinstance(y, DataFrame):
                return y.rename(columns=self.inv_map)
            else:
                return DataFrame(data=y, columns=self.ordered_class_labels_transformed, dtype=np.float32)
        elif isinstance(y, DataFrame):
            return y.to_numpy()
        else:
            return y

    def inverse_transform_proba(self, y, as_pandas=False, as_pred=False):
        y_index = None
        if isinstance(y, DataFrame):
            y_index = y.index
            y = y.to_numpy()
        if self.invalid_class_count > 0:
            y_transformed = np.zeros([len(y), len(self.ordered_class_labels)], dtype=np.float32)
            y_transformed[:, self.label_index_to_keep] = y
        else:
            y_transformed = y
        if as_pred:
            y_transformed = get_pred_from_proba(y_pred_proba=y_transformed, problem_type=MULTICLASS)
            y_transformed = self._convert_to_valid_series(y_transformed)
            y_transformed = y_transformed.map(self.cat_mappings_dependent_var_uncleaned)
            if y_index is not None:
                y_transformed.index = y_index
        if as_pandas and not as_pred:
            y_transformed = DataFrame(data=y_transformed, index=y_index, columns=self.ordered_class_labels, dtype=np.float32)
        return y_transformed

    @staticmethod
    def _generate_categorical_mapping(y: Series) -> dict:
        categories = y.astype('category')
        cat_mappings_dependent_var = dict(enumerate(categories.cat.categories))
        return cat_mappings_dependent_var


# TODO: Expand print statement to multiclass as well
class LabelCleanerBinary(LabelCleaner):
    def __init__(self, y: Series, positive_class=None):
        self.problem_type_transform = BINARY
        y = self._convert_to_valid_series(y)
        self.num_classes = 2
        self.unique_values = list(y.unique())
        if len(self.unique_values) != 2:
            raise AssertionError('y does not contain exactly 2 unique values:', self.unique_values)
        try:
            self.unique_values.sort()
        except TypeError:
            # Contains both str and int
            self.unique_values = list(y.unique())
        # TODO: Clean this code, for loop
        pos_class_warning = None
        if positive_class is not None:
            if positive_class not in self.unique_values:
                raise ValueError(f'positive_class is not a valid class: {self.unique_values} (positive_class={positive_class})')
            negative_class = [c for c in self.unique_values if c != positive_class][0]
            self.inv_map: dict = {negative_class: 0, positive_class: 1}
        elif ((str(False) in [str(val) for val in self.unique_values]) and
              (str(True) in [str(val) for val in self.unique_values])):
            false_val = [val for val in self.unique_values if str(val) == str(False)][0]  # may be str or bool
            true_val = [val for val in self.unique_values if str(val) == str(True)][0]  # may be str or bool
            self.inv_map: dict = {false_val: 0, true_val: 1}
        elif (0 in self.unique_values) and (1 in self.unique_values):
            self.inv_map: dict = {0: 0, 1: 1}
        else:
            self.inv_map: dict = {self.unique_values[0]: 0, self.unique_values[1]: 1}
            pos_class_warning = f'\tNote: For your binary classification, AutoGluon arbitrarily selected which label-value ' \
                                f'represents positive ({self.unique_values[1]}) vs negative ({self.unique_values[0]}) class.\n'\
                                '\tTo explicitly set the positive_class, either rename classes to 1 and 0, or specify positive_class in Predictor init.'
        poslabel = [lbl for lbl in self.inv_map.keys() if self.inv_map[lbl] == 1][0]
        neglabel = [lbl for lbl in self.inv_map.keys() if self.inv_map[lbl] == 0][0]
        logger.log(20, 'Selected class <--> label mapping:  class 1 = %s, class 0 = %s' % (poslabel, neglabel))
        if pos_class_warning is not None:
            logger.log(20, pos_class_warning)
        self.cat_mappings_dependent_var: dict = {v: k for k, v in self.inv_map.items()}
        self.ordered_class_labels_transformed = [0, 1]
        self.ordered_class_labels = [self.cat_mappings_dependent_var[label_transformed] for label_transformed in self.ordered_class_labels_transformed]

    def inverse_transform_proba(self, y, as_pandas=False, as_pred=False):
        if isinstance(y, DataFrame):
            y = copy.deepcopy(y)
            y.columns = copy.deepcopy(self.ordered_class_labels)
            if as_pred:
                y = get_pred_from_proba_df(y, problem_type=self.problem_type_transform)
            if not as_pandas:
                y = y.to_numpy()
        elif as_pred:
            y_index = None
            if isinstance(y, Series):
                y_index = y.index
                y = y.to_numpy()
            y = get_pred_from_proba(y_pred_proba=y, problem_type=self.problem_type_transform)
            y = self._convert_to_valid_series(y)
            y = y.map(self.cat_mappings_dependent_var)
            y = y.to_numpy()
            if as_pandas:
                y = Series(data=y, index=y_index)
        return y

    def _transform(self, y: Series) -> Series:
        y = y.map(self.inv_map)
        return y

    def _inverse_transform(self, y: Series) -> Series:
        return y.map(self.cat_mappings_dependent_var)

    def transform_proba(self, y: Union[DataFrame, Series, np.ndarray], as_pandas=False):
        if not as_pandas and isinstance(y, (Series, DataFrame)):
            y = y.to_numpy()
        elif isinstance(y, DataFrame):
            y = copy.deepcopy(y)
            y.columns = copy.deepcopy(self.ordered_class_labels_transformed)
        return y


class LabelCleanerMulticlassToBinary(LabelCleanerMulticlass):
    def __init__(self, y: Series, y_uncleaned: Series):
        super().__init__(y=y, y_uncleaned=y_uncleaned)
        self.label_cleaner_binary = LabelCleanerBinary(y=y.map(self.inv_map))
        self.problem_type_transform = self.label_cleaner_binary.problem_type_transform

    def _transform(self, y: Series) -> Series:
        y = super()._transform(y)
        y = self.label_cleaner_binary.transform(y)
        return y

    def inverse_transform_proba(self, y, as_pandas=False, as_pred=False):
        if not isinstance(y, DataFrame):
            y = self.convert_binary_proba_to_multiclass_proba(y=y, as_pandas=as_pandas)
        return super().inverse_transform_proba(y, as_pandas=as_pandas, as_pred=as_pred)

    @staticmethod
    def convert_binary_proba_to_multiclass_proba(y, as_pandas=False):
        y_index = None
        if as_pandas and isinstance(y, Series):
            y_index = y.index
        y_transformed = np.zeros([len(y), 2])
        y_transformed[:, 0] = 1 - y
        y_transformed[:, 1] = y
        if as_pandas:
            y_transformed = DataFrame(data=y_transformed, index=y_index)
        return y_transformed


# TODO: Expand functionality if necessary
class LabelCleanerSoftclass(LabelCleaner):
    def __init__(self, y: DataFrame):
        self.problem_type_transform = SOFTCLASS
        self.num_classes = y.shape[1]

    def _transform(self, y: DataFrame) -> DataFrame:
        return y

    def _inverse_transform(self, y: DataFrame) -> DataFrame:
        return y

    @staticmethod
    def _convert_to_valid_series(y: DataFrame) -> DataFrame:
        return y


class LabelCleanerDummy(LabelCleaner):
    def __init__(self, problem_type=REGRESSION):
        self.problem_type_transform = problem_type

    def _transform(self, y: Union[Series, DataFrame]) -> Union[Series, DataFrame]:
        return y

    def _inverse_transform(self, y: Union[Series, DataFrame]) -> Union[Series, DataFrame]:
        return y
