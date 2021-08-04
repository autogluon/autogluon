"""Presets for vision predictors"""
import functools
import warnings
from autogluon.core.utils import get_gpu_free_memory
from autogluon.core import Categorical, Int, Real


def unpack(preset_name):
    if not preset_name in _PRESET_DICTS:
        raise ValueError(f'Unknown preset_name: {preset_name}')
    def _unpack_inner(f):
        @functools.wraps(f)
        def _call(*args, **kwargs):
            gargs, gkwargs = set_presets(preset_name, *args, **kwargs)
            return f(*gargs, **gkwargs)
        return _call
    return _unpack_inner


# Dictionary of preset fit() parameter configurations for ImagePredictor.
preset_image_predictor = dict(
    # Best predictive accuracy with little consideration to inference time or model size. Achieve even better results by specifying a large time_limit value.
    # Recommended for applications that benefit from the best possible model accuracy.
    best_quality={
        'hyperparameters': {
            'model': Categorical('resnet50_v1b', 'resnet101_v1d', 'resnest200'),
            'lr': Real(1e-5, 1e-2, log=True),
            'batch_size': Categorical(8, 16, 32, 64, 128),
            'epochs': 200,
            'early_stop_patience': 50
            },
        'hyperparameter_tune_kwargs': {
            'num_trials': 1024,
            'searcher': 'random',
        },
        'time_limit': 12*3600,
    },

    # Good predictive accuracy with fast inference.
    # Recommended for applications that require reasonable inference speed and/or model size.
    good_quality_fast_inference={
        'hyperparameters': {
            'model': Categorical('resnet50_v1b', 'resnet34_v1b'),
            'lr': Real(1e-4, 1e-2, log=True),
            'batch_size': Categorical(8, 16, 32, 64, 128),
            'epochs': 150,
            'early_stop_patience': 20
            },
        'hyperparameter_tune_kwargs': {
            'num_trials': 512,
            'searcher': 'random',
        },
        'time_limit': 8*3600,
    },

    # Medium predictive accuracy with very fast inference and very fast training time.
    # This is the default preset in AutoGluon, but should generally only be used for quick prototyping.
    medium_quality_faster_train={
        'hyperparameters': {
            'model': 'resnet50_v1b',
            'lr': 0.01,
            'batch_size': 64,
            'epochs': 50,
            'early_stop_patience': 5
            },
        'time_limit': 1*3600,
    },

    # Medium predictive accuracy with very fast inference.
    # Comparing with `medium_quality_faster_train` it uses faster model but explores more hyperparameters.
    medium_quality_faster_inference={
        'hyperparameters': {
            'model': Categorical('resnet18_v1b', 'mobilenetv3_small'),
            'lr': Categorical(0.01, 0.005, 0.001),
            'batch_size': Categorical(64, 128),
            'epochs': Categorical(50, 100),
            'early_stop_patience': 10
            },
        'hyperparameter_tune_kwargs': {
            'num_trials': 32,
            'searcher': 'random',
        },
        'time_limit': 2*3600,
    },
)

# Dictionary of preset fit() parameter configurations for ObjectDetector.
preset_object_detector = dict(
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

)

_PRESET_DICTS = {
    'image_predictor': preset_image_predictor,
    'object_detector': preset_object_detector,
}


def set_presets(preset_name, *args, **kwargs):
    preset_dict = _PRESET_DICTS[preset_name]
    if 'presets' in kwargs:
        presets = kwargs['presets']
        if presets is None:
            return kwargs
        if not isinstance(presets, list):
            presets = [presets]
        preset_kwargs = {}
        for preset in presets:
            if isinstance(preset, str):
                preset_orig = preset
                preset = preset_dict.get(preset, None)
                if preset is None:
                    raise ValueError(f'Preset \'{preset_orig}\' was not found. Valid presets: {list(preset_dict.keys())}')
            if isinstance(preset, dict):
                for key in preset:
                    preset_kwargs[key] = preset[key]
            else:
                raise TypeError(f'Preset of type {type(preset)} was given, but only presets of type [dict, str] are valid.')
        for key in preset_kwargs:
            if key not in kwargs:
                kwargs[key] = preset_kwargs[key]
            elif isinstance(kwargs[key], dict) and isinstance(preset_kwargs[key], dict):
                # allow partially specify dict keys to override default presets
                preset_kwargs[key].update(kwargs[key])
                kwargs[key] = preset_kwargs[key]
    return args, kwargs


def _check_gpu_memory_presets(bs, ngpus_per_trial, min_batch_size=8, threshold=128):
    """Check and report warnings based on free gpu memory.

    Parameters
    ----------
    bs : int or autogluon.core.Space
        Batch size.
    ngpus_per_trial : int
        # gpus per trial
    min_batch_size : int, default = 8
        Minimum batch size to initiate checks, batch size smaller than this will
        not trigger any warning.
    threshold : int, default = 128
        The gpu memory required per sample(unit is MB).

    """
    try:
        if isinstance(bs, Categorical):
            bs = max(bs.data)
        if isinstance(bs, (Real, Int)):
            bs = bs.upper
        if ngpus_per_trial is not None and ngpus_per_trial > 1 and bs > min_batch_size:
            # using gpus, check batch size vs. available gpu memory
            free_gpu_memory = get_gpu_free_memory()
            if not free_gpu_memory:
                warnings.warn('Unable to detect free GPU memory, we are unable to verify '
                              'whether your data mini-batches will fit on the GPU for the specified batch_size.')
            elif len(free_gpu_memory) < ngpus_per_trial:
                warnings.warn(f'Detected GPU memory for {len(free_gpu_memory)} gpus but {ngpus_per_trial} is requested.')
            elif sum(free_gpu_memory[:ngpus_per_trial]) / bs < threshold:
                warnings.warn(f'batch-size: {bs} is potentially larger than what your gpus can fit ' +
                              f'free memory: {free_gpu_memory[:ngpus_per_trial]} ' +
                              'Try reducing "batch_size" if you encounter memory issues')
    except:
        pass
