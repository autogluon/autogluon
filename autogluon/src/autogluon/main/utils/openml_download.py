import os
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

from autogluon.core.utils.files import download

__all__ = ['load_and_split_openml_data']


def load_and_split_openml_data(
        openml_task_id=6, ratio_train_valid=0.33, download_from_openml=False):
    """Download OpenML dataset (task), split of training set

    :param openml_task_id:
    :param ratio_train_valid:
    :param download_from_openml: If False, the data is downloaded from the
        AutoGluon S3 bucket. This works only for tasks which have been
        uploaded there. Use this if the code is run as part of the AutoGluon
        CI system (downloading from OpenML fails too often)
    :return: X_train, X_valid, y_train, y_valid, n_classes
    """
    import openml  # Dependence for tests and docs only
    if not download_from_openml:
        # We download the data from the AutoGluon S3 bucket, avoiding a
        # download from OpenML, which fails too often
        src_url = 'https://autogluon.s3.amazonaws.com/'
        trg_path = './'
        data_path = 'org/openml/www/datasets/{}/'.format(openml_task_id)
        task_path = 'org/openml/www/tasks/{}/'.format(openml_task_id)
        data_files = [
            data_path + x for x in [
                'dataset.arff', 'dataset.pkl.py3', 'description.xml',
                'features.xml', 'qualities.xml']] + [
            task_path + x for x in [
                'datasplits.arff', 'datasplits.pkl.py3', 'task.xml']]
        for fname in data_files:
            trg_fname = trg_path + fname
            if not os.path.exists(trg_fname):
                download(src_url + fname, path=trg_fname)
    # Note: If the files already exist locally, openml will not download them
    openml.config.set_cache_directory("./")
    task = openml.tasks.get_task(openml_task_id)
    n_classes = len(task.class_labels)
    train_indices, test_indices = task.get_train_test_split_indices()
    X, y = task.get_X_and_y()
    # Basic missing values imputation
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    X = imputer.fit_transform(X)
    X_train = X[train_indices]
    y_train = y[train_indices]
    # Train/validation split and standardization of inputs
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train, y_train, random_state=1, test_size=ratio_train_valid)
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    X_train = (X_train - mean) / (std + 1e-10)
    X_valid = (X_valid - mean) / (std + 1e-10)

    return X_train, X_valid, y_train, y_valid, n_classes
