"""Object Detection task"""
import copy
import pickle
import logging
import warnings
import os

from autogluon.core.utils import verbosity2loglevel, get_gpu_count
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
                best_quality={
                    'hyperparameters': {
                        'transfer': Categorical('faster_rcnn_fpn_resnet101_v1d_coco'),
                        'lr': Real(1e-5, 1e-3, log=True),
                        'batch_size': Categorical(4, 8),
                        'epochs': 30
                        },
                    'hyperparameter_tune_kwargs': {
                        'num_trials': 128,
                        'search_strategy': 'bayesopt'},
                    'time_limit': 24*3600,}
                    Best predictive accuracy with little consideration to training/inference time or model size. Achieve even better results by specifying a large time_limit value.
                    Recommended for applications that benefit from the best possible model accuracy and be prepared with the extremly long training time.

                good_quality_fast_inference={
                    'hyperparameters': {
                        'transfer': Categorical('ssd_512_resnet50_v1_coco',
                                                'yolo3_darknet53_coco',
                                                'center_net_resnet50_v1b_coco'),
                        'lr': Real(1e-4, 1e-2, log=True),
                        'batch_size': Categorical(8, 16, 32, 64),
                        'epochs': 50
                        },
                    'hyperparameter_tune_kwargs': {
                        'num_trials': 512,
                        'search_strategy': 'bayesopt'},
                    'time_limit': 12*3600,}
                    Good predictive accuracy with fast inference.
                    Recommended for applications that require reasonable inference speed and/or model size.

                medium_quality_faster_train={
                    'hyperparameters': {
                        'transfer': Categorical('ssd_512_resnet50_v1_coco'),
                        'lr': 0.01,
                        'batch_size': Categorical(8, 16),
                        'epochs': 30
                        },
                    'hyperparameter_tune_kwargs': {
                        'num_trials': 16,
                        'search_strategy': 'random'},
                    'time_limit': 2*3600,}

                    Medium predictive accuracy with very fast inference and very fast training time.
                    This is the default preset in AutoGluon, but should generally only be used for quick prototyping.

                medium_quality_faster_inference={
                    'hyperparameters': {
                        'transfer': Categorical('center_net_resnet18_v1b_coco', 'yolo3_mobilenet1.0_coco'),
                        'lr': Categorical(0.01, 0.005, 0.001),
                        'batch_size': Categorical(32, 64, 128),
                        'epochs': Categorical(30, 50),
                        },
                    'hyperparameter_tune_kwargs': {
                        'num_trials': 32,
                        'search_strategy': 'bayesopt'},
                    'time_limit': 4*3600,}

                    Medium predictive accuracy with very fast inference.
                    Comparing with `medium_quality_faster_train` it uses faster model but explores more hyperparameters.
        hyperparameters : dict, default = None
            Extra hyperparameters for specific models.
            Accepted args includes(not limited to):
            epochs : int, default value based on network
                The `epochs` for model training.
            batch_size : int
                Mini batch size
            lr : float
                Trainer learning rate for optimization process.
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
                search_strategy : str, default = 'random'
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
        search_strategy = kwargs['hyperparameter_tune_kwargs']['search_strategy']
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

        if self._detector is not None:
            self._detector._logger.setLevel(log_level)
            self._detector._logger.propagate = True
            self._fit_summary = self._detector.fit(train_data, tuning_data, 1 - holdout_frac, random_state, resume=False)
            if hasattr(self._classifier, 'fit_history'):
                self._fit_summary['fit_history'] = self._classifier.fit_history()
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
                'search_strategy': search_strategy,
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
        # verbosity
        if log_level > logging.INFO:
            logging.getLogger('gluoncv.auto.tasks.object_detection').propagate = False
            for logger_name in ('SSDEstimator', 'CenterNetEstimator', 'YOLOv3Estimator', 'FasterRCNNEstimator'):
                logging.getLogger(logger_name).setLevel(log_level)
                logging.getLogger(logger_name).propagate = False
        task = _ObjectDetection(config=config)
        task._logger.setLevel(log_level)
        task._logger.propagate = True
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            self._detector = task.fit(train_data, tuning_data, 1 - holdout_frac, random_state)
        self._detector._logger.setLevel(log_level)
        self._detector._logger.propagate = True
        self._fit_summary = task.fit_summary()
        if hasattr(task, 'fit_history'):
            self._fit_summary['fit_history'] = task.fit_history()
        return self

    def _validate_kwargs(self, kwargs):
        """validate and initialize default kwargs"""
        kwargs['holdout_frac'] = kwargs.get('holdout_frac', 0.1)
        if not (0 < kwargs['holdout_frac'] < 1.0):
            raise ValueError(f'Range error for `holdout_frac`, expected to be within range (0, 1), given {kwargs["holdout_frac"]}')
        kwargs['random_state'] = kwargs.get('random_state', None)
        kwargs['nthreads_per_trial'] = kwargs.get('nthreads_per_trial', None)
        kwargs['ngpus_per_trial'] = kwargs.get('ngpus_per_trial', None)
        if kwargs['ngpus_per_trial'] is not None and kwargs['ngpus_per_trial'] > 0:
            detected_gpu = get_gpu_count()
            if detected_gpu < kwargs['ngpus_per_trial']:
                raise ValueError(f"Insufficient detected # gpus {detected_gpu} vs requested {kwargs['ngpus_per_trial']}")
        # tune kwargs
        hpo_tune_args = kwargs.get('hyperparameter_tune_kwargs', {})
        hpo_tune_args['num_trials'] = hpo_tune_args.get('num_trials', 1)
        hpo_tune_args['search_strategy'] = hpo_tune_args.get('search_strategy', 'random')
        if not hpo_tune_args['search_strategy'] in ('random', 'bayesopt', 'grid'):
            raise ValueError(f"Invalid search strategy: {hpo_tune_args['search_strategy']}, supported: ('random', 'bayesopt', 'grid')")
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
            return ret.as_numpy()

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
