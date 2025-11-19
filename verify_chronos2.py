
import pandas as pd
import numpy as np
from autogluon.timeseries import TimeSeriesDataFrame
from autogluon.timeseries.models.chronos import Chronos2Model

def test_chronos2_basic():
    print("Testing basic forecasting...")
    # Create simple dataset
    df = pd.DataFrame({
        "item_id": ["A"] * 20 + ["B"] * 20,
        "timestamp": pd.date_range("2024-01-01", periods=20).tolist() * 2,
        "target": np.random.rand(40)
    })
    data = TimeSeriesDataFrame.from_data_frame(df, id_column="item_id", timestamp_column="timestamp")
    
    model = Chronos2Model(prediction_length=5)
    model.fit(train_data=data)
    predictions = model.predict(data)
    
    print("Predictions shape:", predictions.shape)
    print("Predictions columns:", predictions.columns)
    assert len(predictions) == 10 # 2 items * 5 steps
    assert len(predictions.columns) == 10 # mean + 9 quantiles
    print("Basic forecasting passed!")

def test_chronos2_covariates():
    print("\nTesting covariate support...")
    # Create dataset with covariates
    df = pd.DataFrame({
        "item_id": ["A"] * 30,
        "timestamp": pd.date_range("2024-01-01", periods=30),
        "target": np.concatenate([np.random.rand(25), [np.nan]*5]), # 5 future steps
        "cov1": np.random.rand(30) # known covariate
    })
    
    # Split into past and future for simulation
    past_df = df.iloc[:25].copy()
    future_covariates_df = df.iloc[25:].copy()
    
    data = TimeSeriesDataFrame.from_data_frame(past_df, id_column="item_id", timestamp_column="timestamp")
    known_covariates = TimeSeriesDataFrame.from_data_frame(
        df[["item_id", "timestamp", "cov1"]], 
        id_column="item_id", 
        timestamp_column="timestamp"
    )
    
    model = Chronos2Model(prediction_length=5)
    # Note: fit doesn't really use covariates in the current implementation unless fine-tuning
    model.fit(train_data=data)
    
    predictions = model.predict(data, known_covariates=known_covariates)
    print("Covariate predictions shape:", predictions.shape)
    print("Covariate support passed!")

if __name__ == "__main__":
    test_chronos2_basic()
    test_chronos2_covariates()
