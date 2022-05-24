from autogluon.text.automm.constants import (
    BINARY,
    MULTICLASS,
    REGRESSION,
    ACC,
    RMSE,
    CATEGORICAL,
    NUMERICAL,
)
import numpy as np 
import pandas as pd
import os

FULLNAME = 'full_name'
TYPE = 'problem_type'
METRIC = 'metric'

DATASETS = {
    'ad': {
        FULLNAME: 'adult',
        TYPE: BINARY,
        METRIC: ACC,
    },
    'al': {
        FULLNAME: 'aloi',
        TYPE: MULTICLASS,
        METRIC: ACC,
    },
    'ca': {
        FULLNAME: 'california_housing',
        TYPE: REGRESSION,
        METRIC: RMSE,
    },
    'co': {
        FULLNAME: 'covtype',
        TYPE: MULTICLASS,
        METRIC: ACC,
    },
    'ep': {
        FULLNAME: 'epsilon',
        TYPE: BINARY,
        METRIC: ACC,
    },
    'he': {
        FULLNAME: 'helena',
        TYPE: MULTICLASS,
        METRIC: ACC,
    },
    'hi': {
        FULLNAME: 'higgs_small',
        TYPE: BINARY,
        METRIC: ACC,
    },
    'ja': {
        FULLNAME: 'jannis',
        TYPE: MULTICLASS,
        METRIC: ACC,
    },
    'mi': {
        FULLNAME: 'microsoft',
        TYPE: REGRESSION,
        METRIC: RMSE,
    },
    'ya': {
        FULLNAME: 'yahoo',
        TYPE: REGRESSION,
        METRIC: RMSE,
    },
    'ye': {
        FULLNAME: 'year',
        TYPE: REGRESSION,
        METRIC: RMSE,
    },
}

def load_data_per_split(
    data_path: str,
    mode: str = 'train',
):
    if mode == 'train':
        categorical = 'C_train.npy'
        numerical = 'N_train.npy'
        label = 'y_train.npy'
    elif mode == 'val':
        categorical = 'C_val.npy'
        numerical = 'N_val.npy'
        label = 'y_val.npy'
    elif mode == 'test':
        categorical = 'C_test.npy'
        numerical = 'N_test.npy'
        label = 'y_test.npy'
    else:
        raise ValueError
    
    label_np = np.load(os.path.join(data_path,label))
    label_np = np.expand_dims(label_np,axis=1)

    numerical_np = np.load(os.path.join(data_path,numerical))
    numerical_features = numerical_np.shape[-1]

    np_data = np.append(label_np,numerical_np,axis=1)
    
    categorical_np = None
    if os.path.exists(os.path.join(data_path,categorical)):
        categorical_np = np.load(os.path.join(data_path,categorical))
        categorical_features = categorical_np.shape[-1]
        np_data = np.append(np_data,categorical_np,axis=1)

    pd_columns = [str(i) for i in range(np_data.shape[1])]
    df_data = pd.DataFrame(np_data,columns=pd_columns)

    column_types = {}

    for i in range(1,numerical_features+1,1):
        column_types[str(i)] = NUMERICAL
    if categorical_np is not None:
        for i in range(numerical_features+1, categorical_features+numerical_features+1, 1):
            column_types[str(i)] = CATEGORICAL
    
    return df_data, column_types


def get_tabular_data(
    base_path: str,
    dataset_name: str,
):
    dataset_key = dataset_name.lower()

    assert dataset_key in DATASETS.keys(), f"Not identified data set name {dataset_name}"
    assert os.path.exists(base_path), "No dataset exists"

    data_info = DATASETS[dataset_key]
    data_path = os.path.join(base_path,data_info[FULLNAME])

    problem_type = data_info[TYPE]
    metric = data_info[METRIC]

    train_df, column_types = load_data_per_split(
        data_path = data_path,
        mode = 'train',
    )

    val_df, _ = load_data_per_split(
        data_path = data_path,
        mode = 'val',
    )

    test_df, _  = load_data_per_split(
        data_path = data_path,
        mode = 'test',
    )

    return train_df, val_df, test_df, column_types, problem_type, metric
