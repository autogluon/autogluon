import argparse
from autogluon.tabular import TabularDataset, TabularPredictor
from autogluon.tabular.models import CatBoostModel, KNNModel, LGBModel, XGBoostModel, TabularNeuralNetModel, RFModel
import os
from numpy.core.fromnumeric import trace
import pandas as pd
import traceback

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset_dir', help='path to cpp directory', type=str, default='dataset/cpp')
parser.add_argument('-r', '--result_path', help='file to save test set score to', type=str, default='sanitycheck/cpp/result.csv')
parser.add_argument('-m', '--mode', help='what AutoGluon setting to try', choices=['ag', 'ag-stack'], default='ag')
parser.add_argument('-t', '--time_limit', help='time limit in minutes', type=int, default=60)

args = parser.parse_args()

TIME_LIMIT = args.time_limit * 60.
RESULT_PATH = args.result_path
EXCEPTIONS_PATH = os.path.join(os.path.dirname(args.result_path), 'exceptions.csv')
# DATASETS = [
#     # '0c1b989f-26bc-49c3-832b-0ec1e4b148b1',
#     # '1db99236-0601-4e03-b8bb-96b5eb236d74',
#     # '2cbd9a22-0da1-404d-a7ba-49911840a622',
#     # '3cf28e5f-886a-4ace-bebf-299e1dbde654',
#     '3d18592b-8bc4-4049-8c3e-91c4615eb629',
# ]
DATASETS = os.listdir(args.dataset_dir)
FEATURE_PRUNE_KWARGS = {
    'feature_prune_kwargs': {
        'max_fits': 10,
    }
}


def add_datapoint(result: dict, dataset: str, mode: str, val_score: float, test_score: float, time_limit: float):
    result['dataset'].append(dataset)
    result['mode'].append(mode)
    result['val_score'].append(round(val_score, 4))
    result['test_score'].append(round(test_score, 4))
    result['time_limit'].append(round(time_limit, 4))


def add_exception(exception: dict, dataset: str, type: str, error_str: str, stacktrace: str):
    exception['dataset'].append(dataset)
    exception['type'].append(type)
    exception['error_str'].append(error_str)
    exception['stacktrace'].append(stacktrace)


for dataset in DATASETS:
    train_data = pd.read_csv(os.path.join(args.dataset_dir, dataset, 'train.csv'))
    test_data = pd.merge(pd.read_csv(os.path.join(args.dataset_dir, dataset, 'testFeaturesNoLabel.csv')),
                         pd.read_csv(os.path.join(args.dataset_dir, dataset, 'testLabel.csv')), on='ID')
    y_test = test_data['label']
    presets = ['medium_quality_faster_train'] if args.mode == 'ag' else ['best_quality']

    result = {'dataset': [], 'mode': [], 'val_score': [], 'test_score': [], 'time_limit': []}
    exception = {'dataset': [], 'type': [], 'error_str': [], 'stacktrace': []}
    try:
        predictor = TabularPredictor(label='label')
        predictor = predictor.fit(train_data, presets=presets, time_limit=TIME_LIMIT)
        y_pred = predictor.predict(test_data)
        performance = predictor.evaluate_predictions(y_true=y_test, y_pred=y_pred, auxiliary_metrics=True)
        leaderboard = predictor.leaderboard(test_data)
        best_val_row = leaderboard.loc[leaderboard['score_val'].idxmax()]
        val_score, test_score = best_val_row['score_val'], best_val_row['score_test']
        add_datapoint(result, dataset, presets[0], val_score, test_score, TIME_LIMIT)
    except Exception as e:
        add_exception(exception, dataset, presets[0], str(e), traceback.format_exc())

    try:
        predictor = TabularPredictor(label='label')
        predictor = predictor.fit(train_data, presets=presets, time_limit=TIME_LIMIT, ag_args=FEATURE_PRUNE_KWARGS)
        y_pred = predictor.predict(test_data)
        performance = predictor.evaluate_predictions(y_true=y_test, y_pred=y_pred, auxiliary_metrics=True)
        leaderboard = predictor.leaderboard(test_data)
        best_val_row = leaderboard.loc[leaderboard['score_val'].idxmax()]
        val_score, test_score = best_val_row['score_val'], best_val_row['score_test']
        add_datapoint(result, dataset, presets[0] + "_prune", val_score, test_score, TIME_LIMIT)
    except Exception as e:
        add_exception(exception, dataset, presets[0] + "_prune", str(e), traceback.format_exc())

    result_df = pd.DataFrame(result)
    if os.path.exists(RESULT_PATH):
        original_result_df = pd.read_csv(RESULT_PATH)
        result_df = pd.concat([original_result_df, result_df], axis=0)
    result_df.to_csv(RESULT_PATH, index=False)

    exception_df = pd.DataFrame(exception)
    if os.path.exists(EXCEPTIONS_PATH):
        original_exception_df = pd.read_csv(EXCEPTIONS_PATH)
        exception_df = pd.concat([original_exception_df, exception_df], axis=0)
    exception_df.to_csv(EXCEPTIONS_PATH, index=False)
