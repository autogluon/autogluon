"""Object Detection task"""
import copy
import pickle
import logging
import warnings
import os

import pandas as pd
import numpy as np
from autogluon.core.utils import verbosity2loglevel, get_gpu_count_all
from autogluon.core.utils import set_logger_verbosity
from gluoncv.auto.tasks import ObjectDetection as _ObjectDetection
from ..configs.presets_configs import unpack, _check_gpu_memory_presets

__all__ = ['ObjectDetector']

logger = logging.getLogger()  # return root logger


class ObjectDetector(object):
    """AutoGluon Predictor for detecting objects in images

    Parameters
    ----------
    path : str, default = None
        The directory for saving logs or intermediate data. If unspecified, will create a sub-directory under
        current working directory.
    verbosity : int, default = 2
        Verbosity levels range from 0 to 4 and control how much information is printed.
        Higher levels correspond to more detailed print statements (you can set verbosity = 0 to suppress warnings).
        If using logging, you can alternatively control amount of information printed via logger.setLevel(L),
        where L ranges from 0 to 50 (Note: higher values of L correspond to fewer print statements, opposite of verbosity levels)
    """
    # Dataset is a subclass of `pd.DataFrame`, with `image` and `bbox` columns.
    Dataset = _ObjectDetection.Dataset

    def __init__(self, path=None, verbosity=2):
        if path is None:
            path = os.getcwd()
        self._log_dir = path
        self._verbosity = verbosity
        self._detector = None
        self._fit_summary = {}
        os.makedirs(self._log_dir, exist_ok=True)

    @property
    def path(self):
        return self._log_dir

    @unpack('object_detector')
    def fit(self,
            train_data,
            tuning_data=None,
            time_limit='auto',
            presets=None,
            hyperparameters=None,
            **kwargs):
        """Automatic fit process for object detection.
        Tip: if you observe very slow training speed only happening at the first epoch and your overall time budget
        is not large, you may disable `CUDNN_AUTOTUNE` by setting the environment variable
        `export MXNET_CUDNN_AUTOTUNE_DEFAULT=0` before running your python script or
        insert `import os; os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'` before any code block.
        The tuning is beneficial in terms of training speed in the long run, but may cost your noticeble overhead at
        the begining of each trial.

        Parameters
        ----------
        train_data : pd.DataFrame or str
            Training data, can be a dataframe like image dataset.
            For more details of how to construct a object detection dataset, please checkout:
            `http://preview.d2l.ai/d8/main/object_detection/getting_started.html`.
            If a string is provided, will search for d8 datasets.
        tuning_data : pd.DataFrame or str, default = None
            Holdout tuning data for validation, reserved for model selection and hyperparameter-tuning,
            can be a dataframe like image dataset.
            If a string is provided, will search for k8 datasets.
            If `None`, the validation dataset will be randomly split from `train_data` according to `holdout_frac`.
        time_limit : int, default = 'auto'(defaults to 2 hours if no presets detected)
            Time limit in seconds, if `None`, will run until all tuning and training finished.
            If `time_limit` is hit during `fit`, the
            HPO process will interrupt and return the current best configuration.
        presets : list or str or dict, default = ['medium_quality_faster_train']
            List of preset configurations for various arguments in `fit()`. Can significantly impact predictive accuracy, memory-footprint, and inference latency of trained models,
            and various other properties of the returned `predictor`.
            It is recommended to specify presets and avoid specifying most other `fit()` arguments or model hyperparameters prior to becoming familiar with AutoGluon.
            As an example, to get the most accurate overall predictor (regardless of its efficiency), set `presets='best_quality'`.
            To get good quality with faster inference speed, set `presets='good_quality_faster_inference'`
            Any user-specified arguments in `fit()` will override the values used by presets.
            If specifying a list of presets, later presets will override earlier presets if they alter the same argument.
            For precise definitions of the provided presets, see file: `autogluon/vision/configs/presets_configs.py`.
            Users can specify custom presets by passing in a dictionary of argument values as an element to the list.
            Available Presets: ['best_quality', 'high_quality_fast_inference', 'good_quality_faster_inference', 'medium_quality_faster_train']
            It is recommended to only use one `quality` based preset in a given call to `fit()` as they alter many of the same arguments and are not compatible with each-other.

            Note that depending on your specific hardware limitation(# gpu, size of gpu memory...) your mileage may vary a lot, you may choose lower quality presets if necessary, and
            try to reduce `batch_size` if OOM("RuntimeError: CUDA error: out of memory") happens frequently during the `fit`.

            In-depth Preset Info:
                # Best predictive accuracy with little consideration to inference time or model size. Achieve even better results by specifying a large time_limit value.
                # Recommended for applications that benefit from the best possible model accuracy.
                best_quality={
                    'hyperparameters': {
                        'transfer': 'faster_rcnn_fpn_resnet101_v1d_coco',
                        'lr': Real(1e-5, 1e-3, log=True),
                        'batch_size': Categorical(4, 8),
                        'epochs': 30,
                        'early_stop_patience': 50
                        },
                    'hyperparameter_tune_kwargs': {
                        'num_trials': 128,
                        'searcher': 'random',
                    },
                    'time_limit': 24*3600,
                },

                # Good predictive accuracy with fast inference.
                # Recommended for applications that require reasonable inference speed and/or model size.
                good_quality_fast_inference={
                    'hyperparameters': {
                        'transfer': Categorical('ssd_512_resnet50_v1_coco',
                                                'yolo3_darknet53_coco',
                                                'center_net_resnet50_v1b_coco'),
                        'lr': Real(1e-4, 1e-2, log=True),
                        'batch_size': Categorical(8, 16, 32, 64),
                        'epochs': 50,
                        'early_stop_patience': 20
                        },
                    'hyperparameter_tune_kwargs': {
                        'num_trials': 512,
                        'searcher': 'random',
                    },
                    'time_limit': 12*3600,
                },

                # Medium predictive accuracy with very fast inference and very fast training time.
                # This is the default preset in AutoGluon, but should generally only be used for quick prototyping.
                medium_quality_faster_train={
                    'hyperparameters': {
                        'transfer': 'ssd_512_resnet50_v1_coco',
                        'lr': 0.01,
                        'batch_size': Categorical(8, 16),
                        'epochs': 30,
                        'early_stop_patience': 5
                        },
                    'hyperparameter_tune_kwargs': {
                        'num_trials': 16,
                        'searcher': 'random',
                    },
                    'time_limit': 2*3600,
                },

                # Medium predictive accuracy with very fast inference.
                # Comparing with `medium_quality_faster_train` it uses faster model but explores more hyperparameters.
                medium_quality_faster_inference={
                    'hyperparameters': {
                        'transfer': Categorical('center_net_resnet18_v1b_coco', 'yolo3_mobilenet1.0_coco'),
                        'lr': Categorical(0.01, 0.005, 0.001),
                        'batch_size': Categorical(32, 64, 128),
                        'epochs': Categorical(30, 50),
                        'early_stop_patience': 10
                        },
                    'hyperparameter_tune_kwargs': {
                        'num_trials': 32,
                        'searcher': 'random',
                    },
                    'time_limit': 4*3600,
                },
        hyperparameters : dict, default = None
            Extra hyperparameters for specific models.
            Accepted args includes(not limited to):
            epochs : int, default value based on network
                The `epochs` for model training.
            batch_size : int
                Mini batch size
            lr : float
                Trainer learning rate for optimization process.
            early_stop_patience : int, default=10
                Number of epochs with no improvement after which train is early stopped. Use `None` to disable.
            early_stop_min_delta : float, default=1e-4
                The small delta value to ignore when evaluating the metric. A large delta helps stablize the early
                stopping strategy against tiny fluctuation, e.g. 0.5->0.49->0.48->0.499->0.500001 is still considered as
                a good timing for early stopping.
            early_stop_baseline : float, default=None
                The minimum(baseline) value to trigger early stopping. For example, with `early_stop_baseline=0.5`,
                early stopping won't be triggered if the metric is less than 0.5 even if plateau is detected.
                Use `None` to disable.
            early_stop_max_value : float, default=None
                The max value for metric, early stop training instantly once the max value is achieved. Use `None` to disable.
            You can get the list of accepted hyperparameters in `config.yaml` saved by this predictor.
        **kwargs :
            holdout_frac : float, default = 0.1
                The random split ratio for `tuning_data` if `tuning_data==None`.
            random_state : int, default = None
                The random_state(seed) for shuffling data, only used if `tuning_data==None`.
                Note that the `random_state` only affect the splitting process, not model training.
                If not specified(None), will leave the original random sampling intact.
            nthreads_per_trial : int, default = (# cpu cores)
                Number of CPU threads for each trial, if `None`, will detect the # cores on current instance.
            ngpus_per_trial : int, default = (# gpus)
                Number of GPUs to use for each trial, if `None`, will detect the # gpus on current instance.
            hyperparameter_tune_kwargs: dict, default = None
                num_trials : int, default = 1
                    The limit of HPO trials that can be performed within `time_limit`. The HPO process will be terminated
                    when `num_trials` trials have finished or wall clock `time_limit` is reached, whichever comes first.
                searcher : str, default = 'random'
                    Searcher strategy for HPO, 'random' by default.
                    Options include: ‘random’ (random search), ‘bayesopt’ (Gaussian process Bayesian optimization),
                    ‘grid’ (grid search).
                max_reward : float, default = None
                    The reward threashold for stopping criteria. If `max_reward` is reached during HPO, the scheduler
                    will terminate earlier to reduce time cost.
                scheduler_options : dict, default = None
                    Extra options for HPO scheduler, please refer to :class:`autogluon.core.Searcher` for details.
        """
        # init/validate kwargs
        kwargs = self._validate_kwargs(kwargs)
        # unpack
        num_trials = kwargs['hyperparameter_tune_kwargs']['num_trials']
        nthreads_per_trial = kwargs['nthreads_per_trial']
        ngpus_per_trial = kwargs['ngpus_per_trial']
        holdout_frac = kwargs['holdout_frac']
        random_state = kwargs['random_state']
        scheduler = kwargs['hyperparameter_tune_kwargs']['scheduler']
        searcher = kwargs['hyperparameter_tune_kwargs']['searcher']
        max_reward = kwargs['hyperparameter_tune_kwargs']['max_reward']
        scheduler_options = kwargs['hyperparameter_tune_kwargs']['scheduler_options']

        log_level = verbosity2loglevel(self._verbosity)
        set_logger_verbosity(self._verbosity, logger=logger)
        if presets:
            if not isinstance(presets, list):
                presets = [presets]
            logger.log(20, f'Presets specified: {presets}')

        if time_limit == 'auto':
            # no presets, no user specified time_limit
            time_limit = 7200
            logger.log(20, f'`time_limit=auto` set to `time_limit={time_limit}`.')

        # data sanity check
        train_data = self._validate_data(train_data)
        if tuning_data is not None:
            # FIXME: Use ImagePredictor's tuning_data split logic when None, currently this does not perform an ideal split.
            tuning_data = self._validate_data(tuning_data)

        if self._detector is not None:
            self._detector._logger.setLevel(log_level)
            self._detector._logger.propagate = True
            self._fit_summary = self._detector.fit(train_data, tuning_data, 1 - holdout_frac, random_state, resume=False)
            if hasattr(self._detector, 'fit_history'):
                self._fit_summary['fit_history'] = self._detector.fit_history()
            return self

        # new HPO task
        if time_limit is not None and num_trials is None:
            num_trials = 99999
        if time_limit is None and num_trials is None:
            raise ValueError("`time_limit` and kwargs['hyperparameter_tune_kwargs']['num_trials'] can not be `None` at the same time, "
                             "otherwise the training will not be terminated gracefully.")
        config={'log_dir': self._log_dir,
                'num_trials': 99999 if num_trials is None else max(1, num_trials),
                'time_limits': 2147483647 if time_limit is None else max(1, time_limit),
                'search_strategy': searcher,
                'scheduler': scheduler,
                }
        if max_reward is not None:
            config['max_reward'] = max_reward
        if nthreads_per_trial is not None:
            config['nthreads_per_trial'] = nthreads_per_trial
        if ngpus_per_trial is not None:
            config['ngpus_per_trial'] = ngpus_per_trial
        if isinstance(hyperparameters, dict):
            if 'batch_size' in hyperparameters:
                bs = hyperparameters['batch_size']
                _check_gpu_memory_presets(bs, ngpus_per_trial, 4, 1280)  # 1280MB per sample
            # check if hyperparameters overwriting existing config
            for k, v in hyperparameters.items():
                if k in config:
                    raise ValueError(f'Overwriting {k} = {config[k]} to {v} by hyperparameters is ambiguous.')
            config.update(hyperparameters)
        if scheduler_options is not None:
            config.update(scheduler_options)
        if 'early_stop_patience' not in config:
            config['early_stop_patience'] = 10
        if config['early_stop_patience'] == None:
            config['early_stop_patience'] = -1
        # TODO(zhreshold): expose the transform function(or sign function) for converting custom metrics
        if 'early_stop_baseline' not in config or config['early_stop_baseline'] == None:
            config['early_stop_baseline'] = -np.Inf
        if 'early_stop_max_value' not in config or config['early_stop_max_value'] == None:
            config['early_stop_max_value'] = np.Inf
        # verbosity
        if log_level > logging.INFO:
            logging.getLogger('gluoncv.auto.tasks.object_detection').propagate = False
            for logger_name in ('SSDEstimator', 'CenterNetEstimator', 'YOLOv3Estimator', 'FasterRCNNEstimator'):
                logging.getLogger(logger_name).setLevel(log_level)
                logging.getLogger(logger_name).propagate = False
        task = _ObjectDetection(config=config)
        task.search_strategy = scheduler
        task.scheduler_options['searcher'] = searcher
        task._logger.setLevel(log_level)
        task._logger.propagate = True
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            # TODO: MXNetErrorCatcher was removed because it didn't return traceback,
            #  Re-add once it returns full traceback regardless of which exception was caught
            self._detector = task.fit(train_data, tuning_data, 1 - holdout_frac, random_state)
        self._detector._logger.setLevel(log_level)
        self._detector._logger.propagate = True
        self._fit_summary = task.fit_summary()
        if hasattr(task, 'fit_history'):
            self._fit_summary['fit_history'] = task.fit_history()
        return self

    def _validate_data(self, data):
        """Check whether data is valid, try to convert with best effort if not"""
        if len(data) < 1:
            raise ValueError('Empty dataset.')
        if not (hasattr(data, 'classes') and hasattr(data, 'to_mxnet')):
            if isinstance(data, pd.DataFrame):
                # raw dataframe, try to add metadata automatically
                infer_classes = []
                if 'image' in data.columns:
                    # check image relative/abs path is valid
                    sample = data.iloc[0]['image']
                    if not os.path.isfile(sample):
                        raise OSError(f'Detected invalid image path `{sample}`, please ensure all image paths are absolute or you are using the right working directory.')
                if 'rois' in data.columns and 'image' in data.columns:
                    sample = data.iloc[0]['rois']
                    for sample_key in ('class', 'xmin', 'ymin', 'xmax', 'ymax'):
                        assert sample_key in sample, f'key `{sample_key}` required in `rois`'
                    class_column = data.rois.apply(lambda x: x.get('class', 'unknown'))
                    infer_classes = class_column.unique().tolist()
                    data['rois'] = data['rois'].apply(lambda x: x.update({'difficult': x.get('difficult', 0)} or x))
                    data = _ObjectDetection.Dataset(data.sort_values('image').reset_index(drop=True), classes=infer_classes)
                elif 'image' in data and 'class' in data and 'xmin' in data and 'ymin' in data and 'xmax' in data and 'ymax' in data:
                    infer_classes = data['class'].unique().tolist()
                    if 'difficult' not in data.columns:
                        data['difficult'] = 0
                    data = _ObjectDetection.Dataset(data.sort_values('image').reset_index(drop=True), classes=infer_classes)
                    data = data.pack()
                    data.classes = infer_classes
                else:
                    err_msg = 'Unable to convert raw DataFrame to ObjectDetector Dataset, ' + \
                              '`image` and `rois` columns are required.' + \
                              'You may visit `https://auto.gluon.ai/stable/tutorials/object_detection/dataset.html` ' + \
                              'for details.'
                    raise AttributeError(err_msg)
                logger.log(20, 'Converting raw DataFrame to ObjectDetector.Dataset...')
                logger.log(20, f'Detected {len(infer_classes)} unique classes: {infer_classes}')
                instruction = 'train_data = ObjectDetector.Dataset(train_data, classes=["foo", "bar"])'
                logger.log(20, f'If you feel the `classes` is inaccurate, please construct the dataset explicitly, e.g. {instruction}')
        return data

    def _validate_kwargs(self, kwargs):
        """validate and initialize default kwargs"""

        valid_kwargs = {'holdout_frac', 'random_state', 'nthreads_per_trial', 'ngpus_per_trial', 'hyperparameter_tune_kwargs'}
        invalid_kwargs = []
        for key in kwargs:
            if key not in valid_kwargs:
                invalid_kwargs.append(key)
        if invalid_kwargs:
            raise KeyError(f'Invalid kwargs specified: {invalid_kwargs}. Valid kwargs names: {list(valid_kwargs)}')

        kwargs['holdout_frac'] = kwargs.get('holdout_frac', 0.1)
        if not (0 < kwargs['holdout_frac'] < 1.0):
            raise ValueError(f'Range error for `holdout_frac`, expected to be within range (0, 1), given {kwargs["holdout_frac"]}')
        kwargs['random_state'] = kwargs.get('random_state', None)
        kwargs['nthreads_per_trial'] = kwargs.get('nthreads_per_trial', None)
        kwargs['ngpus_per_trial'] = kwargs.get('ngpus_per_trial', None)
        if kwargs['ngpus_per_trial'] is not None and kwargs['ngpus_per_trial'] > 0:
            detected_gpu = self._get_num_gpus_available()
            if detected_gpu < kwargs['ngpus_per_trial']:
                raise ValueError(f"Insufficient detected # gpus {detected_gpu} vs requested {kwargs['ngpus_per_trial']}")
        # tune kwargs
        hpo_tune_args = kwargs.get('hyperparameter_tune_kwargs', {})
        hpo_tune_args['num_trials'] = hpo_tune_args.get('num_trials', 1)
        hpo_tune_args['searcher'] = hpo_tune_args.get('searcher', 'random')
        if not hpo_tune_args['searcher'] in ('random', 'bayesopt', 'grid'):
            raise ValueError(f"Invalid searcher: {hpo_tune_args['searcher']}, supported: ('random', 'bayesopt', 'grid')")
        hpo_tune_args['scheduler'] = hpo_tune_args.get('scheduler', 'local')
        if not hpo_tune_args['scheduler'] in ('fifo', 'local'):
            raise ValueError(f"Invalid searcher: {hpo_tune_args['searcher']}, supported: ('fifo', 'local')")
        hpo_tune_args['max_reward'] = hpo_tune_args.get('max_reward', None)
        if hpo_tune_args['max_reward'] is not None and hpo_tune_args['max_reward'] < 0:
            raise ValueError(f"Expected `max_reward` to be a positive float number between 0 and 1.0, given {hpo_tune_args['max_reward']}")
        hpo_tune_args['scheduler_options'] = hpo_tune_args.get('scheduler_options', None)
        kwargs['hyperparameter_tune_kwargs'] = hpo_tune_args
        return kwargs

    def predict(self, data, as_pandas=True):
        """Predict objects in image, return the confidences, bounding boxes of each predicted object.

        Parameters
        ----------
        data : str, pd.DataFrame or ndarray
            The input data, can be str(filepath), pd.DataFrame with 'image' column, or raw ndarray input.
        as_pandas : bool, default = True
            Whether to return the output as a pandas object (True) or list of numpy array(s) (False).
            Pandas object is a DataFrame.

        Returns
        -------

        pd.DataFrame
            The returned dataframe will contain (`pred_score`, `pred_bbox`, `pred_id`).
            If more than one image in input, the returned dataframe will contain `images` column,
            and all results are concatenated.
        """
        if self._detector is None:
            raise RuntimeError('Detector is not initialized, try `fit` first.')
        ret = self._detector.predict(data)
        if as_pandas:
            return ret
        else:
            return ret.to_numpy()

    def evaluate(self, data):
        """Evaluate model performance on validation data.

        Parameters
        ----------
        data : pd.DataFrame or iterator
            The validation data.
        """
        if self._detector is None:
            raise RuntimeError('Detector not initialized, try `fit` first.')
        return self._detector.evaluate(data)

    def fit_summary(self):
        """Return summary of last `fit` process.

        Returns
        -------
        dict
            The summary of last `fit` process. Major keys are ('train_map', 'val_map', 'total_time',...)

        """
        return copy.copy(self._fit_summary)

    def save(self, path=None):
        """Dump predictor to disk.

        Parameters
        ----------
        path : str
            The file name of saved copy. If not specified(None), will automatically save to `self.path` directory
            with filename `object_detector.ag`

        """
        if path is None:
            path = os.path.join(self.path, 'object_detector.ag')
        with open(path, 'wb') as fid:
            pickle.dump(self, fid)

    @classmethod
    def load(cls, path, verbosity=2):
        """Load previously saved predictor.

        Parameters
        ----------
        path : str
            The file name for saved pickle file. If `path` is a directory, will try to load the file `object_detector.ag` in
            this directory.
        verbosity : int, default = 2
            Verbosity levels range from 0 to 4 and control how much information is printed.
            Higher levels correspond to more detailed print statements (you can set verbosity = 0 to suppress warnings).
            If using logging, you can alternatively control amount of information printed via logger.setLevel(L),
            where L ranges from 0 to 50 (Note: higher values of L correspond to fewer print statements, opposite of verbosity levels)

        """
        if os.path.isdir(path):
            path = os.path.join(path, 'object_detector.ag')
        with open(path, 'rb') as fid:
            obj = pickle.load(fid)
        obj._verbosity = verbosity
        return obj

    @staticmethod
    def _get_num_gpus_available():
        return get_gpu_count_all()
