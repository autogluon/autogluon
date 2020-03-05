import gc
import os

import numpy as np
from pandas import DataFrame, Series

from autogluon import try_import_lightgbm
from autogluon.utils.tabular.utils.decorators import calculate_time
from autogluon.utils.tabular.utils.savers import save_pd
from ...constants import MULTICLASS
from ...utils import logger


def func_generator(metric, is_higher_better, needs_pred_proba, problem_type):
    if needs_pred_proba:
        if problem_type == MULTICLASS:
            def function_template(y_hat, data):
                y_true = data.get_label()
                y_hat = y_hat.reshape(len(np.unique(y_true)), -1).T
                return metric.name, metric(y_true, y_hat), is_higher_better
        else:
            def function_template(y_hat, data):
                y_true = data.get_label()
                return metric.name, metric(y_true, y_hat), is_higher_better
    else:
        if problem_type == MULTICLASS:
            def function_template(y_hat, data):
                y_true = data.get_label()
                y_hat = y_hat.reshape(len(np.unique(y_true)), -1)
                y_hat = y_hat.argmax(axis=0)
                return metric.name, metric(y_true, y_hat), is_higher_better
        else:
            def function_template(y_hat, data):
                y_true = data.get_label()
                y_hat = np.round(y_hat)
                return metric.name, metric(y_true, y_hat), is_higher_better
    return function_template


def construct_dataset(x: DataFrame, y: Series, location=None, reference=None, params=None, save=False, weight=None):
    try_import_lightgbm()
    import lightgbm as lgb

    dataset = lgb.Dataset(data=x, label=y, reference=reference, free_raw_data=True, params=params, weight=weight)

    if save:
        assert location is not None
        saving_path = f'{location}.bin'
        if os.path.exists(saving_path):
            os.remove(saving_path)

        os.makedirs(os.path.dirname(saving_path), exist_ok=True)
        dataset.save_binary(saving_path)
        # dataset_binary = lgb.Dataset(location + '.bin', reference=reference, free_raw_data=False)# .construct()

    return dataset


# TODO: not used, remove ?
def construct_dataset_low_memory(X: DataFrame, y: Series, location, reference=None, params=None):
    try_import_lightgbm()
    import lightgbm as lgb
    cat_columns = list(X.select_dtypes(include='category').columns.values)
    # X = X.drop(columns_categorical, axis=1)

    X[cat_columns] = X[cat_columns].apply(lambda x: x.cat.codes)

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

        columns = columns + [x for x in _temp.columns]

        nodel_ind = [_temp.columns.tolist().index(x) for x in _temp.columns]

        _temp = _temp.iloc[:split_train, nodel_ind]

        ei = _temp.values.shape[1]
        mmap[:, si:si + ei] = _temp.values
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

    # use_columns = columns
    # muse_columns = [columns.index(x) for x in use_columns]

    # d_train = _train[:, muse_columns]
    xgtrain = lgb.Dataset(_train, label=y, params=params, reference=reference, categorical_feature=cat_columns, feature_name=columns)
    # d_val = _val[:, muse_columns]
    # xgvalid = lgb.Dataset(d_val, label=y_test, reference=xgtrain, **params)

    # bst = lgb.train(model_params, xgtrain, valid_sets=[xgvalid], valid_names=['valid'], evals_result=evals_results, **fit_params)

    return xgtrain


# TODO: not used, remove ?
@calculate_time
def construct_dataset_lowest_memory(X: DataFrame, y: Series, location, reference=None, params=None):
    try_import_lightgbm()
    import lightgbm as lgb

    cat_columns = list(X.select_dtypes(include='category').columns.values)
    X[cat_columns] = X[cat_columns].apply(lambda x: x.cat.codes)

    columns = list(X.columns)
    cat_columns_index = [columns.index(cat) for cat in cat_columns]

    saving_path = f'{location}.csv'
    logger.log(15, f'Saving... {saving_path}')
    save_pd.save(path=saving_path, df=X, header=False, index=True)

    xgtrain = lgb.Dataset(saving_path, label=y, params=params,
                          reference=reference, categorical_feature=cat_columns_index,
                          feature_name=columns)

    return xgtrain
