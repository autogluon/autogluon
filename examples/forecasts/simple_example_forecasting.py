from autogluon.utils.forecasting.ml.models.mqcnn.mqcnn_model import MQCNNModel
from autogluon.task.forecasting.dataset import ForecastingDataset
import autogluon as ag


dataset = ForecastingDataset(
    train_path="./COV19/processed_train.csv",
    test_path="./COV19/processed_test.csv",
    index_column="name",
    target_column="ConfirmedCases",
    date_column="Date")
# model = MQCNNModel(hyperparameters={"context_length": 10})
# path = "test_model"
# model.fit(dataset.train_ds)
# model.save(path)
# reloaded_model = MQCNNModel.load(path)
# # predicted_test = model.predict(dataset.test_ds)
# score = reloaded_model.score(dataset.test_ds)


@ag.args(context_length=ag.space.Int(1, 20))
def train_fn(args, reporter):
    context_length = args.context_length
    train_model = MQCNNModel(hyperparameters={"context_length": context_length})
    train_model.fit(dataset.train_ds)
    for e in range(3):
        train_score = train_model.score(dataset.test_ds)
        reporter(epoch=e + 1, wQuantileLoss=train_score, context_length=context_length)


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
