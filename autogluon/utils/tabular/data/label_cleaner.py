import logging
from pandas import Series
import numpy as np

from ..ml.constants import BINARY, MULTICLASS, REGRESSION

logger = logging.getLogger(__name__)


# LabelCleaner cleans labels prior to entering feature generation
class LabelCleaner:
    @staticmethod
    def construct(problem_type: str, y: Series, y_uncleaned: Series):
        if problem_type == BINARY:
            return LabelCleanerBinary(y)
        elif problem_type == MULTICLASS:
            if len(y.unique()) == 2:
                return LabelCleanerMulticlassToBinary(y, y_uncleaned)
            else:
                return LabelCleanerMulticlass(y, y_uncleaned)
        elif problem_type == REGRESSION:
            return LabelCleanerDummy()
        else:
            raise NotImplementedError

    def transform(self, y: Series) -> Series:
        raise NotImplementedError

    def inverse_transform(self, y: Series) -> Series:
        raise NotImplementedError

    def transform_proba(self, y):
        return y

    def inverse_transform_proba(self, y):
        return y


class LabelCleanerMulticlass(LabelCleaner):
    def __init__(self, y: Series, y_uncleaned: Series):
        self.cat_mappings_dependent_var: dict = self._generate_categorical_mapping(y)
        self.inv_map: dict = {v: k for k, v in self.cat_mappings_dependent_var.items()}

        self.cat_mappings_dependent_var_uncleaned: dict = self._generate_categorical_mapping(y_uncleaned)
        self.inv_map_uncleaned: dict = {v: k for k, v in self.cat_mappings_dependent_var_uncleaned.items()}

        self.num_classes = len(self.cat_mappings_dependent_var.keys())
        self.ordered_class_labels = list(y_uncleaned.astype('category').cat.categories)
        self.valid_ordered_class_labels = list(y.astype('category').cat.categories)
        self.invalid_class_count = len(self.ordered_class_labels) - len(self.valid_ordered_class_labels)
        self.labels_to_zero_fill = [1 if label not in self.valid_ordered_class_labels else 0 for label in self.ordered_class_labels]
        self.label_index_to_keep = [i for i, label in enumerate(self.labels_to_zero_fill) if label == 0]
        self.label_index_to_remove = [i for i, label in enumerate(self.labels_to_zero_fill) if label == 1]

    def transform(self, y: Series) -> Series:
        if type(y) == np.ndarray:
            y = Series(y)
        y = y.map(self.inv_map)
        return y

    def inverse_transform(self, y: Series) -> Series:
        y = y.map(self.cat_mappings_dependent_var)
        return y

    # TODO: Unused?
    def transform_proba(self, y):
        if self.invalid_class_count > 0:
            # This assumes y has only 0's for any columns it is about to remove, if it does not, weird things may start to happen since rows will not sum to 1
            y_transformed = np.delete(y, self.label_index_to_remove, axis=1)
            return y_transformed
        else:
            return y

    def inverse_transform_proba(self, y):
        if self.invalid_class_count > 0:
            y_transformed = np.zeros([len(y), len(self.ordered_class_labels)])
            y_transformed[:, self.label_index_to_keep] = y
            return y_transformed
        else:
            return y

    def _generate_categorical_mapping(self, y: Series) -> dict:
        categories = y.astype('category')
        cat_mappings_dependent_var = dict(enumerate(categories.cat.categories))
        return cat_mappings_dependent_var


# TODO: Expand print statement to multiclass as well
class LabelCleanerBinary(LabelCleaner):
    def __init__(self, y: Series):
        self.num_classes = 2
        self.unique_values = list(y.unique())
        if len(self.unique_values) != 2:
            raise AssertionError('y does not contain exactly 2 unique values:', self.unique_values)
        # TODO: Clean this code, for loop
        if (1 in self.unique_values) and (2 in self.unique_values):
            self.inv_map: dict = {1: 0, 2: 1}
        elif ('1' in self.unique_values) and ('2' in self.unique_values):
            self.inv_map: dict = {'1': 0, '2': 1}
        elif (False in self.unique_values) and (True in self.unique_values):
            self.inv_map: dict = {False: 0, True: 1}
        elif (0 in self.unique_values) and (1 in self.unique_values):
            self.inv_map: dict = {0: 0, 1: 1}
        elif ('0' in self.unique_values) and ('1' in self.unique_values):
            self.inv_map: dict = {'0': 0, '1': 1}
        elif ('False' in self.unique_values) and ('True' in self.unique_values):
            self.inv_map = {'False': 0, 'True': 1}
        elif ('No' in self.unique_values) and ('Yes' in self.unique_values):
            self.inv_map: dict = {'No': 0, 'Yes': 1}
        elif ('N' in self.unique_values) and ('Y' in self.unique_values):
            self.inv_map: dict = {'N': 0, 'Y': 1}
        elif ('n' in self.unique_values) and ('y' in self.unique_values):
            self.inv_map: dict = {'n': 0, 'y': 1}
        elif ('F' in self.unique_values) and ('T' in self.unique_values):
            self.inv_map: dict = {'F': 0, 'T': 1}
        elif ('f' in self.unique_values) and ('t' in self.unique_values):
            self.inv_map: dict = {'f': 0, 't': 1}
        else:
            self.inv_map: dict = {self.unique_values[0]: 0, self.unique_values[1]: 1}
            logger.log(15, 'Note: For your binary classification, AutoGluon arbitrarily selects which label-value represents positive vs negative class')
        poslabel = [lbl for lbl in self.inv_map.keys() if self.inv_map[lbl] == 1][0]
        neglabel = [lbl for lbl in self.inv_map.keys() if self.inv_map[lbl] == 0][0]
        logger.log(20, 'Selected class <--> label mapping:  class 1 = %s, class 0 = %s' % (poslabel, neglabel))
        self.cat_mappings_dependent_var: dict = {v: k for k, v in self.inv_map.items()}

    def transform(self, y: Series) -> Series:
        if type(y) == np.ndarray:
            y = Series(y)
        y = y.map(self.inv_map)
        return y

    def inverse_transform(self, y: Series) -> Series:
        y = y.map(self.cat_mappings_dependent_var)
        return y


class LabelCleanerMulticlassToBinary(LabelCleanerMulticlass):
    def __init__(self, y: Series, y_uncleaned: Series):
        super().__init__(y=y, y_uncleaned=y_uncleaned)
        self.label_cleaner_binary = LabelCleanerBinary(y=y.map(self.inv_map))

    def transform(self, y: Series) -> Series:
        y = super().transform(y)
        y = self.label_cleaner_binary.transform(y)
        return y

    def inverse_transform_proba(self, y):
        y = self.convert_binary_proba_to_multiclass_proba(y=y)
        return super().inverse_transform_proba(y)

    @staticmethod
    def convert_binary_proba_to_multiclass_proba(y):
        y_transformed = np.zeros([len(y), 2])
        y_transformed[:, 0] = 1 - y
        y_transformed[:, 1] = y
        return y_transformed


class LabelCleanerDummy(LabelCleaner):
    def __init__(self):
        self.num_classes = None

    def transform(self, y: Series) -> Series:
        if type(y) == np.ndarray:
            y = Series(y)
        return y

    def inverse_transform(self, y: Series) -> Series:
        return y
