
import functools


def unpack(g):
    def _unpack_inner(f):
        @functools.wraps(f)
        def _call(*args, **kwargs):
            gargs, gkwargs = g(*args, **kwargs)
            return f(*gargs, **gkwargs)
        return _call
    return _unpack_inner


# Dictionary of preset fit() parameter configurations.
preset_dict = dict(
    # Best predictive accuracy with little consideration to inference time or disk usage. Achieve even better results by specifying a large time_limit value.
    # Recommended for applications that benefit from the best possible model accuracy.
    best_quality={'auto_stack': True},

    # Identical to best_quality but additionally trains refit_full models that have slightly lower predictive accuracy but are over 10x faster during inference and require 10x less disk space.
    best_quality_with_high_quality_refit={'auto_stack': True, 'refit_full': True},

    # High predictive accuracy with fast inference. ~10x-200x faster inference and ~10x-200x lower disk usage than `best_quality`.
    # Recommended for applications that require reasonable inference speed and/or model size.
    high_quality_fast_inference_only_refit={'auto_stack': True, 'refit_full': True, 'set_best_to_refit_full': True, 'save_bagged_folds': False},

    # Good predictive accuracy with very fast inference. ~4x faster inference and ~4x lower disk usage than `high_quality_fast_inference_only_refit`.
    # Recommended for applications that require fast inference speed.
    good_quality_faster_inference_only_refit={'auto_stack': True, 'refit_full': True, 'set_best_to_refit_full': True, 'save_bagged_folds': False, 'hyperparameters': 'light'},

    # Medium predictive accuracy with very fast inference and very fast training time. ~20x faster training than `good_quality_faster_inference_only_refit`.
    # This is the default preset in AutoGluon, but should generally only be used for quick prototyping, as `good_quality_faster_inference_only_refit` results in significantly better predictive accuracy and faster inference time.
    medium_quality_faster_train={'auto_stack': False},

    # Optimizes result immediately for deployment by deleting unused models and removing training artifacts.
    # Often can reduce disk usage by ~2-4x with no negatives to model accuracy or inference speed.
    # This will disable numerous advanced functionality, but has no impact on inference.
    # Recommended for applications where the inner details of AutoGluon's training is not important and there is no intention of manually choosing between the final models.
    # This preset pairs well with the other presets such as `good_quality_faster_inference_only_refit` to make a very compact final model.
    # Identical to calling `predictor.delete_models(models_to_keep='best', dry_run=False)` and `predictor.save_space()` directly after `fit()`.
    optimize_for_deployment={'keep_only_best': True, 'save_space': True},

    # Disables automated feature generation when text features are detected.
    # This is useful to determine how beneficial text features are to the end result, as well as to ensure features are not mistaken for text when they are not.
    ignore_text={'_feature_generator_kwargs': {'enable_text_ngram_features': False, 'enable_text_special_features': False}},

    # TODO: Consider HPO-enabled configs if training time doesn't matter but inference latency does.
)


def set_presets(*args, **kwargs):
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
    return args, kwargs
