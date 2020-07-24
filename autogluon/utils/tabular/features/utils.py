import logging

from pandas import Series

logger = logging.getLogger(__name__)


def check_if_useless_feature(X: Series) -> bool:
    if len(X.unique()) <= 1:
        return True
    else:
        return False
