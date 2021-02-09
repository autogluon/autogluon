"""Presets for vision predictors"""
import functools
from autogluon.core import Categorical, Real


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
            'epochs': 200
            },
        'hyperparameter_tune_kwargs': {
            'num_trials': 1024,
            'search_strategy': 'bayesopt'
        }
    },

    # Good predictive accuracy with fast inference.
    # Recommended for applications that require reasonable inference speed and/or model size.
    good_quality_fast_inference={
        'hyperparameters': {
            'model': Categorical('resnet50_v1b', 'resnet34_v1b'),
            'lr': Real(1e-4, 1e-2, log=True),
            'batch_size': Categorical(8, 16, 32, 64, 128),
            'epochs': 150
            },
        'hyperparameter_tune_kwargs': {
            'num_trials': 512,
            'search_strategy': 'bayesopt'
        }
    },

    # Medium predictive accuracy with very fast inference and very fast training time. 
    # This is the default preset in AutoGluon, but should generally only be used for quick prototyping.
    medium_quality_faster_train={
        'hyperparameters': {
            'model': 'resnet50_v1b',
            'lr': 0.01,
            'batch_size': 64,
            'epochs': 50
            },
        'hyperparameter_tune_kwargs': {
            'num_trials': 8,
            'search_strategy': 'random'
        }
    },

    # Medium predictive accuracy with very fast inference.
    # Comparing with `medium_quality_faster_train` it uses faster model but explores more hyperparameters.
    medium_quality_faster_inference={
        'hyperparameters': {
            'model': Categorical('resnet18_v1b', 'mobilenetv3_small'),
            'lr': Categorical(0.01, 0.005, 0.001),
            'batch_size': Categorical(64, 128),
            'epochs': Categorical(50, 100),
            },
        'hyperparameter_tune_kwargs': {
            'num_trials': 32,
            'search_strategy': 'bayesopt'
        }
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
            'epochs': 30
            },
        'hyperparameter_tune_kwargs': {
            'num_trials': 128,
            'search_strategy': 'bayesopt'
        }
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
            'epochs': 50
            },
        'hyperparameter_tune_kwargs': {
            'num_trials': 512,
            'search_strategy': 'bayesopt'
        }
    },

    # Medium predictive accuracy with very fast inference and very fast training time. 
    # This is the default preset in AutoGluon, but should generally only be used for quick prototyping.
    medium_quality_faster_train={
        'hyperparameters': {
            'transfer': 'ssd_512_resnet50_v1_coco',
            'lr': 0.01,
            'batch_size': Categorical(8, 16),
            'epochs': 30
            },
        'hyperparameter_tune_kwargs': {
            'num_trials': 16,
            'search_strategy': 'random'
        }
    },

    # Medium predictive accuracy with very fast inference.
    # Comparing with `medium_quality_faster_train` it uses faster model but explores more hyperparameters.
    medium_quality_faster_inference={
        'hyperparameters': {
            'transfer': Categorical('center_net_resnet18_v1b_coco', 'yolo3_mobilenet1.0_coco'),
            'lr': Categorical(0.01, 0.005, 0.001),
            'batch_size': Categorical(32, 64, 128),
            'epochs': Categorical(30, 50),
            },
        'hyperparameter_tune_kwargs': {
            'num_trials': 32,
            'search_strategy': 'bayesopt'
        }
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
