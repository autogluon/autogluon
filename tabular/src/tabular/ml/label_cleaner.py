from pandas import Series
from tabular.ml.constants import BINARY, MULTICLASS, REGRESSION, LANGUAGE_MODEL


# LabelCleaner cleans labels prior to entering feature generation
class LabelCleaner:
    @staticmethod
    def construct(problem_type: str, y: Series):
        if problem_type == BINARY:
            return LabelCleanerMulticlass(y) # Even for binary tasks, no guarantee y is {0, 1} so need to transform.
            # return LabelCleanerDummy()
        elif problem_type == MULTICLASS:
            return LabelCleanerMulticlass(y)
        elif problem_type == REGRESSION:
            return LabelCleanerDummy()
        elif problem_type == LANGUAGE_MODEL:
            return LabelCleanerDummy()
        else:
            raise NotImplementedError

    def transform(self, y: Series) -> Series:
        raise NotImplementedError

    def inverse_transform(self, y: Series) -> Series:
        raise NotImplementedError


class LabelCleanerMulticlass(LabelCleaner):
    def __init__(self, y: Series):
        self.cat_mappings_dependent_var: dict = self._generate_categorical_mapping(y)
        self.inv_map: dict = {v: k for k, v in self.cat_mappings_dependent_var.items()}
        self.num_classes = len(self.cat_mappings_dependent_var.keys())

    def transform(self, y: Series) -> Series:
        y = y.map(self.inv_map)
        return y

    def inverse_transform(self, y: Series) -> Series:
        y = y.map(self.cat_mappings_dependent_var)
        return y

    def _generate_categorical_mapping(self, y: Series) -> dict:
        categories = y.astype('category')
        cat_mappings_dependent_var = dict(enumerate(categories.cat.categories))
        return cat_mappings_dependent_var


class LabelCleanerDummy(LabelCleaner):
    def __init__(self):
        self.num_classes = None

    def transform(self, y: Series) -> Series:
        return y

    def inverse_transform(self, y: Series) -> Series:
        return y
