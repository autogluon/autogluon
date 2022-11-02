import numpy as np
import pandas as pd

import autogluon.eda.auto as auto
from autogluon.eda.analysis import MissingValuesAnalysis


def test_MissingValuesAnalysis():
    cols = list("AB")
    df_train = pd.DataFrame((np.arange(100))[:, None].repeat([len(cols)], axis=1), columns=cols)
    df_test = pd.DataFrame((np.arange(200))[:, None].repeat([len(cols)], axis=1), columns=cols)
    for df in [df_train, df_test]:
        df["A"] = (df["A"] % 4).replace(2, np.NaN)

    state = auto.analyze(
        train_data=df_train, test_data=df_test, return_state=True, anlz_facets=[MissingValuesAnalysis()]
    )

    assert state.missing_statistics.train_data.count == {"A": 25}
    assert state.missing_statistics.train_data.ratio == {"A": 0.25}
    assert state.missing_statistics.train_data.data is df_train

    assert state.missing_statistics.test_data.count == {"A": 50}
    assert state.missing_statistics.test_data.ratio == {"A": 0.25}
    assert state.missing_statistics.test_data.data is df_test
