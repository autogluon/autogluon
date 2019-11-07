import pandas as pd
import numpy as np
import os
from pandas import DataFrame, Series
import lightgbm as lgb
import gc
from tabular.ml.constants import BINARY, MULTICLASS, REGRESSION
from tabular.utils.savers import save_pd
from tabular.utils.decorators import calculate_time
from matplotlib import pyplot as plt
import itertools


def get_pred_from_proba(y_pred_proba, problem_type=BINARY):
    if problem_type == BINARY:
        y_pred = [1 if pred >= 0.5 else 0 for pred in y_pred_proba]
    elif problem_type == REGRESSION:
        y_pred = y_pred_proba
    else:
        y_pred = np.argmax(y_pred_proba, axis=1)
    return y_pred


def construct_dataset(x: DataFrame, y: Series, location=None, reference=None, params=None, save=False, weight=None):
    # save_pd.save(path=location + '.csv', df=x, header=False)
    feature_list = list(x.columns.values)
    # dataset = lgb.Dataset(data=location + '.csv', label=y, reference=reference, feature_name=feature_list)
    dataset = lgb.Dataset(data=x, label=y, reference=reference, free_raw_data=True, params=params, weight=weight)

    if save:
        if os.path.exists(location + '.bin'):
            os.remove(location + '.bin')
        else:
            pass

        os.makedirs(os.path.dirname(location + '.bin'), exist_ok=True)
        dataset.save_binary(location + '.bin')
        # dataset_binary = lgb.Dataset(location + '.bin', reference=reference, free_raw_data=False)# .construct()


    return dataset


def construct_dataset_low_memory(X: DataFrame, y: Series, location, reference=None, params=None):
    cat_columns = list(X.select_dtypes(include='category').columns.values)
    # X = X.drop(columns_categorical, axis=1)

    X[cat_columns] = X[cat_columns].apply(lambda x: x.cat.codes)

    columns = list(X.columns)
    for column in columns:
        column_data = X[column]

    split_train = len(X)
    # split_train = 11111111  # border between train/test in pickle (length of our train)
    n_attrs = len(X.columns)
    # n_attrs = 25  # as is

    pickle_list = [X]
    # pickle_list = ['attrs_xxxx', 'attrs_base_cnt', 'attrs_nunique']  # list of pickled attrs
    # del_cols = ['click_time', 'day', ]  # attrs to be deleted in final train


    si = 0
    os.makedirs(os.path.dirname(location + '.mmp'), exist_ok=True)
    mmap = np.memmap(location + '.mmp', dtype='float32', mode='w+', shape=(split_train, n_attrs))


    columns = []
    for pkl in pickle_list:
        _temp = pkl
        # _temp = load_attrs(pkl)

        _columns = [x for x in _temp.columns]
        columns = columns + _columns

        nodel_ind = [_temp.columns.tolist().index(x) for x in _temp.columns]

        _temp = _temp.iloc[:split_train, nodel_ind]

        ei = _temp.values.shape[1]
        mmap[:, si:si+ei] = _temp.values
        si += ei

        del _temp
        gc.collect()

    mmap.flush()
    del mmap
    gc.collect()

    mmap = np.memmap(location + '.mmp', dtype='float32', mode='r', shape=(split_train, n_attrs))
    _train = np.array(mmap[:split_train])
    # _val = np.array(mmap[split_train:])

    # _train = _train[:, columns.index('is_attributed')]
    # _val = _val[:, columns.index('is_attributed')]

    use_columns = columns
    # muse_columns = [columns.index(x) for x in use_columns]

    # d_train = _train[:, muse_columns]
    xgtrain = lgb.Dataset(_train, label=y, params=params, reference=reference, categorical_feature=cat_columns, feature_name=columns)
    # d_val = _val[:, muse_columns]
    # xgvalid = lgb.Dataset(d_val, label=y_test, reference=xgtrain, **params)

    # bst = lgb.train(model_params, xgtrain, valid_sets=[xgvalid], valid_names=['valid'], evals_result=evals_results, **fit_params)

    return xgtrain


@calculate_time
def construct_dataset_lowest_memory(X: DataFrame, y: Series, location, reference=None, params=None):

    cat_columns = list(X.select_dtypes(include='category').columns.values)

    columns = list(X.columns)
    X[cat_columns] = X[cat_columns].apply(lambda x: x.cat.codes)

    cat_columns_index = [columns.index(cat) for cat in cat_columns]


    # X['_y_'] = y

    print('saving...', location + '.csv')
    save_pd.save(path=location + '.csv', df=X, header=False, index=True)

    xgtrain = lgb.Dataset(location + '.csv', label=y, params=params, reference=reference, categorical_feature=cat_columns_index,
                          feature_name=columns,
                          )

    return xgtrain


def load_attrs(fname, data_dir='./data/'):
    fname = data_dir + fname + '.pkl'
    print('loading {}... '.format(fname))
    return pd.read_pickle(fname)


def convert_categorical_to_int(X):
    X = X.copy()
    cat_columns = X.select_dtypes(['category']).columns
    X[cat_columns] = X[cat_columns].apply(lambda x: x.cat.codes)
    return X

def plot_confusion_matrix(confusion_matrix, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        confusion_matrix = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(confusion_matrix, interpolation='nearest', cmap=cmap)
    plt.title(title)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = confusion_matrix.max() / 2.
    for i, j in itertools.product(range(confusion_matrix.shape[0]), range(confusion_matrix.shape[1])):
        plt.text(j, i, format(confusion_matrix[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if confusion_matrix[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
