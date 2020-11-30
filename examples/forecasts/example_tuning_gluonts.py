from autogluon.forecasting.models.gluonts_model.mqcnn.mqcnn_model import MQCNNModel
from autogluon.forecasting.task.forecasting.dataset import TimeSeriesDataset

import autogluon.core as ag
from autogluon.core.scheduler import FIFOScheduler

dataset = TimeSeriesDataset(
    train_path="./COV19/processed_train.csv",
    test_path="./COV19/processed_test.csv",
    index_column="name",
    target_column="ConfirmedCases",
    time_column="Date")

eval_metric = 'mean_wQuantileLoss'


@ag.args(context_length=ag.Int(1, 20))
def train_fn(args, reporter):
    context_length = args.context_length
    model = MQCNNModel(path="model/",
                       freq=dataset.freq,
                       prediction_length=dataset.prediction_length,
                       hyperparameters={"context_length": context_length, "epochs": 10})
    train_data = dataset.train_data
    val_data = dataset.test_data
    model.fit(train_data)
    for e in range(3):
        val_score = model.score(val_data, eval_metric)
        reporter(epoch=e + 1, validation_score=-val_score, context_length=context_length)


def basic_hp():
    scheduler = ag.scheduler.FIFOScheduler(train_fn,
                                           searcher="random",
                                           resource={'num_cpus': 1, 'num_gpus': 0},
                                           time_out=10,
                                           reward_attr="validation_score",
                                           time_attr='epoch')
    scheduler.run()
    scheduler.join_jobs()
    scheduler.get_training_curves(plot=True)
    print(scheduler.get_best_config(), scheduler.get_best_reward())


def advanced_hp():
    model = MQCNNModel(path="model/",
                       freq=dataset.freq,
                       prediction_length=dataset.prediction_length,
                       hyperparameters={"context_length": ag.Int(1, 20), "epochs": 10, "num_batches_per_epoch": 10})
    scheduler_options = FIFOScheduler, {"searcher": "random",
                                        "resource": {"num_cpus": 1, "num_gpus": 0},
                                        "num_trials": 10,
                                        "reward_attr": "validation_performance",
                                        "time_attr": "epoch",
                                        "time_out": 1000}
    train_data = dataset.train_data
    val_data = dataset.test_data
    results = model.hyperparameter_tune(train_data=train_data, val_data=val_data, scheduler_options=scheduler_options)
    print(results)


basic_hp()
# advanced_hp()
