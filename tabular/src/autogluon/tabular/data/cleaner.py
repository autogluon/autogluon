import logging
from pandas import DataFrame

from ..constants import BINARY, MULTICLASS, REGRESSION

logger = logging.getLogger(__name__)


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
        else:
            raise NotImplementedError

    def fit(self, X: DataFrame) -> DataFrame:
        raise NotImplementedError

    def fit_transform(self, X: DataFrame) -> DataFrame:
        self.fit(X)
        return self.transform(X)

    def transform(self, X: DataFrame) -> DataFrame:
        raise NotImplementedError


class CleanerDummy(Cleaner):
    def __init__(self):
        pass

    def fit(self, X: DataFrame) -> DataFrame:
        pass

    def transform(self, X: DataFrame) -> DataFrame:
        return X


class CleanerMulticlass(Cleaner):
    def __init__(self, label: str, threshold: int):
        self.label = label
        self.threshold = threshold
        self.valid_classes = None

    def fit(self, X: DataFrame):
        self.valid_classes = self.get_valid_classes(X=X, label=self.label, threshold=self.threshold)

    def transform(self, X: DataFrame) -> DataFrame:
        return self.remove_classes(X=X, label=self.label, valid_classes=self.valid_classes)

    @staticmethod
    def get_valid_classes(X, label, threshold):
        class_counts = X[label].value_counts()
        class_counts_valid = class_counts[class_counts >= threshold]
        valid_classes = list(class_counts_valid.index)
        sum_prior = sum(class_counts)
        sum_after = sum(class_counts_valid)
        percent = sum_after / sum_prior
        if len(valid_classes) < len(class_counts):
            logger.log(25, 'Warning: Some classes in the training set have fewer than %s examples. AutoGluon will only keep %s out of %s classes for training and will not try to predict the rare classes. '
                           'To keep more classes, increase the number of datapoints from these rare classes in the training data or reduce label_count_threshold.' % (threshold, len(valid_classes), len(class_counts)))
        if percent < 1.0:
            logger.log(25, 'Fraction of data from classes with at least %s examples that will be kept for training models: %s' % (threshold, percent))
        return valid_classes

    @staticmethod
    def remove_classes(X, label, valid_classes):
        X = X[X[label].isin(valid_classes)]
        return X
