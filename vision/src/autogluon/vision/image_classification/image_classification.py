"""Image classification task"""
import copy
import pickle
import logging

import pandas as pd
from gluoncv.auto.tasks import ImageClassification as _ImageClassification
from gluoncv.model_zoo import get_model_list

__all__ = ['ImageClassification']

class ImageClassification(object):
    """AutoGluon Predictor for classifying images based on their whole contents

    Parameters
    ----------
    config : dict
        The configurations, can be nested dict.
    logger : logging.Logger
        The desired logger object, use `None` for module specific logger with default setting.
    net : mx.gluon.Block
        The custom network. If defined, the model name in config will be ignored so your
        custom network will be used for training rather than pulling it from model zoo.
    """
    # Dataset is a subclass of `pd.DataFrame`, with `image` and `label` columns.
    Dataset = _ImageClassification.Dataset

    def __init__(self, log_dir=None):
        self._log_dir = log_dir
        self._classifier = None
        self._fit_summary = {}

    def fit(self,
            train_data,
            val_data=None,
            train_size=0.9,
            random_state=None,
            time_limit=12*60*60,
            epochs=None,
            num_trials=None,
            hyperparameters=None,
            search_strategy='random',
            scheduler_options=None,
            nthreads_per_trial=None,
            ngpus_per_trial=None,
            dist_ip_addrs=None):
        """Automatic fit process.

        Parameters
        ----------
        train_data : pd.DataFrame or str
            Training data, can be a dataframe like image dataset.
            If a string is provided, will search for k8 datasets.
        val_data : pd.DataFrame or str
            Training data, can be a dataframe like image dataset.
            If a string is provided, will search for k8 datasets.
            If `None`, the validation dataset will be randomly split from `train_data`.
        train_size : float
            The random split ratio for `train_data` if `val_data==None`.
            The new `val_data` size will be `1-train_size`.
        random_state : numpy.random.state
            The random_state for shuffling, only used if `val_data==None`.
            Note that the `random_state` only affect the splitting process, not model training.
        time_limit : int
            Time limit in seconds, default is 12 hours. If `time_limit` is hit during `fit`, the
            HPO process will interupt and return the current best configuration.
        epochs : int
            The `epochs` for model training, if `None` is provided, then default `epochs` for model
            will be used.
        num_trials : int, default is 1
            The number of HPO trials. If `None`, will run only one trial.
        hyperparameters : dict
            Extra hyperparameters for specific models.
        search_strategy : str
            Searcher strategy for HPO, 'random' by default.
        scheduler_options : dict
            Extra options for HPO scheduler, please refer to `autogluon.Searcher` for details.
        nthreads_per_trial : int
            Number of CPU threads for each trial, if `None`, will detect the # cores on current instance.
        ngpus_per_trial : int
            Number of GPUs to use for each trial, if `None`, will detect the # gpus on current instance.
        dist_ip_addrs : list
            If not `None`, will spawn tasks on distributed nodes.

        """
        use_rec = False
        if isinstance(train_data, str) and train_data == 'imagenet':
            logging.warn('ImageNet is a huge dataset which cannot be downloaded directly, ' +
                         'please follow the data preparation tutorial in GluonCV.' +
                         'The following record files(symlinks) will be used: \n' +
                         'rec_train : ~/.mxnet/datasets/imagenet/rec/train.rec\n' +
                         'rec_train_idx : ~/.mxnet/datasets/imagenet/rec/train.idx\n' +
                         'rec_val : ~/.mxnet/datasets/imagenet/rec/val.rec\n' +
                         'rec_val_idx : ~/.mxnet/datasets/imagenet/rec/val.idx\n')
            train_data = pd.DataFrame({'image': [], 'label': []})
            val_data = pd.DataFrame({'image': [], 'label': []})
            use_rec = True
        if isinstance(train_data, str):
            from d8.image_classification import Dataset as D8D
            names = D8D.list()
            if train_data.lower() in names:
                train_data = D8D.get(train_data)
            if val_data is None:
                train_data, val_data = train_data.split(train_size)
        if isinstance(val_data, str):
            from d8.image_classification import Dataset as D8D
            names = D8D.list()
            if val_data.lower() in names:
                val_data = D8D.get(val_data)
        if self._classifier is not None:
            self._fit_summary = self._classifier.fit(train_data, val_data, train_size, random_state, resume=False)
            return

        # new HPO task
        config={'log_dir': self._log_dir,
                'num_trials': 1 if num_trials is None else num_trials,
                'time_limits': time_limit,
                'search_strategy': search_strategy,
                }
        if nthreads_per_trial is not None:
            config.update({'nthreads_per_trial': nthreads_per_trial})
        if ngpus_per_trial is not None:
            config.update({'ngpus_per_trial': ngpus_per_trial})
        if dist_ip_addrs is not None:
            config.update({'dist_ip_addrs': dist_ip_addrs})
        if epochs is not None:
            config.update({'epochs': epochs})
        if isinstance(hyperparameters, dict):
            net = hyperparameters.pop('net', None)
            if net is not None:
                config.update({'custom_net': net})
            optimizer = hyperparameters.pop('optimizer', None)
            if optimizer is not None:
                config.update({'custom_optimizer': optimizer})
            config.update(hyperparameters)
        if scheduler_options is not None:
            config.update(scheduler_options)
        if use_rec == True:
            config['use_rec'] = True
        task = _ImageClassification(config=config)
        self._classifier = task.fit(train_data, val_data, train_size, random_state)
        self._fit_summary = task.fit_summary()
        return self

    def predict(self, x):
        """Predict images as a whole, return the probabilities of each category.

        Parameters
        ----------
        x : str, pd.DataFrame or ndarray
            The input, can be str(filepath), pd.DataFrame with 'image' column, or raw ndarray input.

        Returns
        -------

        pd.DataFrame
            The returned dataframe will contain probs of each category. If more than one image in input,
            the returned dataframe will contain `images` column, and all results are concatenated.
        """
        if self._classifier is None:
            raise RuntimeError('Classifier is not initialized, try `fit` first.')
        return self._classifier.predict(x)

    def predict_feature(self, x):
        """Predict images visual feature representations, return the features as numpy (1xD) vector.

        Parameters
        ----------
        x : str, pd.DataFrame or ndarray
            The input, can be str(filepath), pd.DataFrame with 'image' column, or raw ndarray input.

        Returns
        -------

        pd.DataFrame
            The returned dataframe will contain image features. If more than one image in input,
            the returned dataframe will contain `images` column, and all results are concatenated.
        """
        if self._classifier is None:
            raise RuntimeError('Classifier is not initialized, try `fit` first.')
        return self._classifier.predict_feature(x)

    def evaluate(self, val_data):
        """Evaluate model performance on validation data.

        Parameters
        ----------
        val_data : pd.DataFrame or iterator
            The validation data.
        """
        if self._classifier is None:
            raise RuntimeError('Classifier not initialized, try `fit` first.')
        return self._classifier.evaluate(val_data)

    def fit_summary(self):
        """Return summary of last `fit` process.

        Returns
        -------
        dict
            The summary of last `fit` process. Major keys are ('train_acc', 'val_acc', 'total_time',...)

        """
        return copy.copy(self._fit_summary)

    def save(self, file_name):
        """Dump predictor to disk.

        Parameters
        ----------
        file_name : str
            The file name of saved copy.

        """
        with open(file_name, 'wb') as fid:
            pickle.dump(self, fid)

    @classmethod
    def load(cls, file_name):
        """Load previously saved predictor.

        Parameters
        ----------
        file_name : str
            The file name for saved pickle file.

        """
        with open(file_name, 'rb') as fid:
            obj = pickle.load(fid)
        return obj

    @classmethod
    def list_models(cls):
        """Get the list of supported model names in model zoo that
        can be used for image classification.

        Returns
        -------
        tuple of str
            A tuple of supported model names in str.

        """
        return tuple(_SUPPORTED_MODELS)


def _get_supported_models():
    all_models = get_model_list()
    blacklist = ['ssd', 'faster_rcnn', 'mask_rcnn', 'fcn', 'deeplab',
                 'psp', 'icnet', 'fastscnn', 'danet', 'yolo', 'pose',
                 'center_net', 'siamrpn', 'monodepth',
                 'ucf101', 'kinetics', 'voc', 'coco', 'citys', 'mhpv1',
                 'ade', 'hmdb51', 'sthsth', 'otb']
    cls_models = [m for m in all_models if not any(x in m for x in blacklist)]
    return cls_models

_SUPPORTED_MODELS = _get_supported_models()
