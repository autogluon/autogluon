import numpy as np
import ConfigSpace as CS
from ray import tune

from .task.image_classification import pipeline


def fit(data,
        nets,
        optimizers=None,
        metrics=None,
        losses=None,
        searcher=None,
        trial_scheduler=None,
        resume=False,
        savedir='./outputdir/',
        visualizer='tensorboard',
        stop_criterion={'time_limits': 1 * 60 * 60,
                        'max_metric': 0.80,
                        'max_trial_count': 100},
        resources_per_trial={'max_num_gpus': 1,
                             'max_num_cpus': 4,
                             'max_training_epochs': 2},
        *args):
    cs = CS.ConfigurationSpace()
    assert data is not None
    assert nets is not None
    if data.search_space is not None:
        cs.add_configuration_space(data.search_space)
    if nets.search_space is not None:
        cs.add_configuration_space(nets.search_space)
    if optimizers.search_space is not None:
        cs.add_configuration_space(optimizers.search_space)
    if metrics.search_space is not None:
        cs.add_configuration_space(metrics.search_space)
    if losses.search_space is not None:
        cs.add_configuration_space(losses.search_space)
    import json
    with open('config_space.json', 'w') as f:
        f.write(json.write(cs))
    with open('config_space.json') as f:
        search_space = json.load(f)

    if searcher is None:
        searcher = tune.automl.search_policy.RandomSearch(search_space,
                                                          stop_criterion['max_metric'],
                                                          stop_criterion['max_trial_count'])
    if trial_scheduler is None:
        trial_scheduler = tune.schedulers.FIFOScheduler()

    tune.register_trainable(
        "TRAIN_FN", lambda config, reporter: pipeline.train_image_classification(
            args, config, reporter))
    trials = tune.run(
        "TRAIN_FN",
        name=args.expname,
        verbose=2,
        scheduler=trial_scheduler,
        **{
            "stop": {
                "mean_accuracy": stop_criterion['max_metric'],
                "training_iteration": resources_per_trial['max_training_epochs']
            },
            "resources_per_trial": {
                "cpu": int(resources_per_trial['max_num_cpus']),
                "gpu": int(resources_per_trial['max_num_gpus'])
            },
            "num_samples": resources_per_trial['max_trial_count'],
            "config": {
                "lr": tune.sample_from(lambda spec: np.power(
                    10.0, np.random.uniform(-4, -1))),
                "momentum": tune.sample_from(lambda spec: np.random.uniform(
                    0.85, 0.95)),
            }
        })
    best_result = max([trial.best_result for trial in trials])
    return trials, best_result, cs
