from forecasting.task.forecasting.forecasting import Forecasting as task

from forecasting.task.forecasting.dataset import TimeSeriesDataset
import core as ag
import matplotlib.pyplot as plt
from forecasting.utils.ml.models.mqcnn.mqcnn_model import MQCNNModel

dataset = TimeSeriesDataset(
    train_path="./COV19/processed_test.csv",
    # test_path="./COV19/processed_test.csv",
    prediction_length=19,
    index_column="name",
    target_column="ConfirmedCases",
    time_column="Date")


metric = "MAPE"
predictor = task.fit(hyperparameters={"mqcnn": {"context_length": ag.Int(80, 120),
                                                "epochs": 3,
                                                "num_batches_per_epoch": 32,
                                                "freq": dataset.freq,
                                                "prediction_length": dataset.prediction_length}},
                     hyperparameter_tune=True,
                     metric=metric,
                     train_ds=dataset.train_ds,
                     test_ds=dataset.test_ds)
# forecasts, tss = predictor.predict(dataset.test_ds)
# print(forecasts)
print(predictor.best_configs())
print(predictor.evaluate(dataset.test_ds))
# plots
scores = []
x_axis = []
for i in range(10):
    path = "./model/mqcnn/" + f"trial_{i}.pkl"
    model = MQCNNModel.load(path)
    score = model.score(dataset.test_ds, metric=metric)
    context_length = model.params["context_length"]
    plt.plot([0, 1], [score, score], label=f"trial_{i}, cl={context_length}")
    plt.ylabel(metric)
    # scores.append(score)
    # x_axis.append(f"trial_{i}, cl={context_length}")
# plt.plot(x_axis, scores, "o")
plt.legend()
plt.show()
