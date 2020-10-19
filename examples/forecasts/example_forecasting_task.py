from autogluon.task.forecasting.forecasting import Forecasting as task
from autogluon.task.forecasting.dataset import ForecastingDataset
import autogluon as ag
import matplotlib.pyplot as plt
from autogluon.utils.forecasting.ml.models.mqcnn.mqcnn_model import MQCNNModel

dataset = ForecastingDataset(
    train_path="./COV19/processed_test.csv",
    # test_path="./COV19/processed_test.csv",
    prediction_length=19,
    index_column="name",
    target_column="ConfirmedCases",
    date_column="Date")


predictor = task.fit(hyperparameters={"mqcnn": {"context_length": ag.Int(80, 120), "epochs": 3, "num_batches_per_epoch": 32}},
                     hyperparameter_tune=True, train_ds=dataset.train_ds, test_ds=dataset.test_ds)
# forecasts, tss = predictor.predict(dataset.test_ds)
# print(forecasts)
print(predictor.best_configs())
print(predictor.evaluate(dataset.test_ds))
# plots
scores = []
x_axis = []
for i in range(10):
    path = "./model/mqcnn/" + f"trial_{i}"
    model = MQCNNModel.load(path)
    score = model.score(dataset.test_ds)
    context_length = model.params["context_length"]
    plt.plot([0, 1], [score, score], label=f"trial_{i}, cl={context_length}")
    # scores.append(score)
    # x_axis.append(f"trial_{i}, cl={context_length}")
# plt.plot(x_axis, scores, "o")
plt.legend()
plt.show()
