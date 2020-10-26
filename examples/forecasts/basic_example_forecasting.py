from forecasting.utils.ml.models.mqcnn.mqcnn_model import MQCNNModel
from forecasting.task.forecasting.dataset import TimeSeriesDataset
import core as ag


dataset = TimeSeriesDataset(
    train_path="./COV19/processed_train.csv",
    test_path="./COV19/processed_test.csv",
    index_column="name",
    target_column="ConfirmedCases",
    time_column="Date")
# model = MQCNNModel(hyperparameters={"context_length": 5 * dataset.prediction_length,
#                                     "prediction_length": dataset.prediction_length,
#                                     "freq": dataset.freq})
# path = "test_model"
# model.fit(dataset.train_ds)
# # predicted_test = model.predict(dataset.test_ds)
# score = model.score(dataset.test_ds)
# print(score)


@ag.args(context_length=ag.Int(1, 20))
def train_fn(args, reporter):
    context_length = args.context_length
    train_model = MQCNNModel(hyperparameters={"context_length": context_length})
    train_model.fit(dataset.train_ds)
    for e in range(3):
        train_score = train_model.score(dataset.test_ds)
        reporter(epoch=e + 1, wQuantileLoss=train_score, context_length=context_length)


def basic_hp():
    scheduler = ag.scheduler.FIFOScheduler(train_fn,
                                           searcher="random",
                                           resource={'num_cpus': 1, 'num_gpus': 0},
                                           num_trials=5,
                                           reward_attr='wQuantileLoss',
                                           time_attr='epoch')
    scheduler.run()
    scheduler.join_jobs()
    scheduler.get_training_curves(plot=True)
    print(scheduler.get_best_config(), scheduler.get_best_reward())
    print(scheduler.training_history)


# basic_hp()
def advanced_hp():
    model = MQCNNModel(hyperparameters={"context_length": ag.Int(1, 20)})
    results = model.hyperparameter_tune(train_data=dataset.train_ds, test_data=dataset.test_ds)
    print(results)


# basic_hp()
advanced_hp()
