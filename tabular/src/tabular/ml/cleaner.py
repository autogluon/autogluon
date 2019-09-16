from pandas import DataFrame
from tabular.ml.constants import BINARY, MULTICLASS, REGRESSION, LANGUAGE_MODEL


# Cleaner cleans data prior to entering feature generation
class Cleaner:
    @staticmethod
    def construct(problem_type: str, label: str, threshold: int):
        if problem_type == BINARY:
            return CleanerDummy()
        elif problem_type == MULTICLASS:
            return CleanerMulticlass(label=label, threshold=threshold)
        elif problem_type == REGRESSION:
            return CleanerDummy()
        elif problem_type == LANGUAGE_MODEL:
            return CleanerDummy()
        else:
            raise NotImplementedError

    def clean(self, X: DataFrame) -> DataFrame:
        raise NotImplementedError


class CleanerDummy(Cleaner):
    def __init__(self):
        pass

    def clean(self, X: DataFrame) -> DataFrame:
        return X


class CleanerMulticlass(Cleaner):
    def __init__(self, label: str, threshold: int):
        self.label = label
        self.threshold = threshold

    def clean(self, X: DataFrame) -> DataFrame:
        X = self.remove_rare_classes(X=X, label=self.label, threshold=self.threshold)
        return X

    @staticmethod
    def remove_rare_classes(X, label, threshold):
        class_counts = X[label].value_counts()

        class_counts_counts = class_counts / sum(class_counts)

        class_counts_valid = class_counts[class_counts > threshold]

        valid_classes = list(class_counts_valid.index)

        X = X[X[label].isin(valid_classes)]

        sum_prior = sum(class_counts)
        sum_after = sum(class_counts_valid)

        percent = sum_after / sum_prior

        print('classes kept:', len(valid_classes), '/', len(class_counts))
        print('percent of data kept:', percent)

        return X