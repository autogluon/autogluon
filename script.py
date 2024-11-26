import pandas as pd

from autogluon.tabular import TabularPredictor


if __name__ == '__main__':
    
    train_data = pd.read_csv('https://autogluon.s3.amazonaws.com/datasets/Inc/train.csv')
    subsample_size = 500  # subsample data for faster demo, try setting this to much larger values
    if subsample_size is not None and subsample_size < len(train_data):
        train_data = train_data.sample(n=subsample_size, random_state=0)
    test_data = pd.read_csv('https://autogluon.s3.amazonaws.com/datasets/Inc/test.csv')

    path_weights_regressor = '/home/ubuntu/tabular/TabForestPFN/outputs/mix_8_regression_quantile_uniform/weights/model_step_600000.pt'

    tabpfnmix_default = {
        "weights_path_regressor": path_weights_regressor,
        "n_ensembles": 1,
        "max_epochs": 3,
    }

    hyperparameters = {
        "TABPFNMIX": [
            tabpfnmix_default,
        ],
    }

    label = "age"
    problem_type = "regression"

    predictor = TabularPredictor(
        label=label,
        problem_type=problem_type,
    )
    predictor = predictor.fit(
        train_data=train_data,
        hyperparameters=hyperparameters,
        verbosity=3,
    )

    predictor.leaderboard(test_data, display=True)