import random
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split


def make_dataset(request, seed):
    TEST_SIZE = 0.5
    # Ensure our datasets and model calls remain deterministic.
    random.seed(seed)
    np.random.seed(seed)
    if request['type'] == 'regression':

        x, y = make_regression(n_samples=int(request['n_samples'] * (1 / (1 - TEST_SIZE))),
                               n_features=request['n_features'],
                               noise=4)  # To make it hard enough that we get better performance on slower models
    elif request['type'] == 'classification':
        x, y = make_classification(n_samples=int(request['n_samples'] * (1 / (1 - TEST_SIZE))),
                                   n_features=request['n_features'],
                                   n_informative=request['n_informative'],
                                   n_redundant=request['n_classes'] - request['n_informative'],
                                   n_classes=request['n_classes'],
                                   class_sep=0.4)  # To make it hard enough that we get better performance on slower models
    else:
        assert False, "Unrecognised request type '{request['type'}'"

    dfx = pd.DataFrame(x)
    dfy = pd.DataFrame(y, columns=['label'])

    # Make some columns categorical if required.
    if request['n_categorical'] > 0:
        cols_to_convert = random.sample(set(dfx.columns.values), k=request['n_categorical'])
        for col in cols_to_convert:
            dfx[col] = dfx[col].astype(int)
            vals = np.unique(dfx[col])
            # Shuffle the categoricals so there's no pattern in their ordering.
            vals2 = vals.copy() - min(vals)
            np.random.shuffle(vals2)
            mapper = dict(zip(vals, vals2))
            dfx[col] = dfx[col].map(mapper)
            dfx[col] = dfx[col].astype("category")

    x_train, x_test, y_train, y_test = train_test_split(dfx, dfy, test_size=TEST_SIZE)
    dftrain = pd.concat([x_train, y_train], axis=1)
    dftest = pd.concat([x_test, y_test], axis=1)

    return (dftrain, dftest)