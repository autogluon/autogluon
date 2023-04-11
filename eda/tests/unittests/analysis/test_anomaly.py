import numpy as np
import pandas as pd
from pyod.models.lof import LOF

from autogluon.eda.analysis import AnomalyDetector


def test_AnomalyDetector():
    np.random.seed(0)
    cols = list("AB")
    df = pd.DataFrame(np.random.randint(0, 2, size=(1000, 2)), columns=cols)
    df["label"] = [0, 1] * int(len(df) / 2)
    idx_anomaly = [8, 12, 25, 22, 76, 48]
    df.loc[idx_anomaly, cols] = 5
    ad = AnomalyDetector(label="label", n_folds=2, detector_list=[LOF(n_neighbors=5), LOF(n_neighbors=10)])
    anomaly_scores = ad.fit_transform(df)
    assert sorted(anomaly_scores.sort_values(ascending=False).index[: len(idx_anomaly)]) == sorted(idx_anomaly)
    assert ad.transform(pd.DataFrame({"A": [0], "B": [1]}))[0] < 1
    assert ad.transform(pd.DataFrame({"A": [5], "B": [10]}))[0] > 1
