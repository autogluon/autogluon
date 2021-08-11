"""
Show that permutation feature importance score based pruning applied on LightGBM and
NNFastAiTabularModel can a) improve test score and b) remove synthetic features. Compare
model accuracies for original datasets and datasets with added noise columns. Try both
bagged and non bagged versions. Datasets are adult, airlines, australian, covertype, and
higgs.
"""

import argparse
import os
import pandas as pd
from autogluon.core.models import BaggedEnsembleModel
from autogluon.tabular.models import LGBModel, NNFastAiTabularModel
from autogluon.core.data import LabelCleaner
from autogluon.core.utils import infer_problem_type
from autogluon.core.utils.feature_selection import FeatureSelector, add_noise_column
from autogluon.features.generators import AutoMLPipelineFeatureGenerator
from sklearn.model_selection import train_test_split
from typing import Sequence


parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data_dir', help='dataset directory', type=str, required=True)
parser.add_argument('-l', '--label', help='label column name', type=str, default='class')
parser.add_argument('-r', '--result_path', help='file to save test set score to', type=str, default='sanitycheck/algorithm_prune/result.csv')
parser.add_argument('-t', '--time_limit', help='time limit models have to train in seconds', type=int, default=3600)
parser.add_argument('-m', '--max_fits', help='maximum times a model can be fit during pruning', type=int, default=10)
parser.add_argument('-q', '--stop_threshold', help='how many iteration to allow even if score does not increase', type=int, default=10)
parser.add_argument('-p', '--prune_threshold', help='pruning threshold for feature importance', type=float, default=None)
parser.add_argument('-s', '--seed', help='number of seeds to evaluate', type=int, default=1)
args = parser.parse_args()


MODELS = [LGBModel]
DATA_DIR = args.data_dir
RESULT_PATH = args.result_path
SEEDS = args.seed
DATA_NAME = os.path.basename(DATA_DIR)
TASK_TYPES = ['original', 'normal_easy', 'normal_hard']
TRAIN_NAMES = ['train_data.csv', 'n1_r0.2_train_data.csv', 'n1_r1.0_train_data.csv']
TEST_NAMES = ['test_data.csv', 'n1_r0.2_test_data.csv', 'n1_r1.0_test_data.csv']


def add_datapoint(result: dict, dataset: str, task_type: str, model: str, seed: int, features: Sequence[str], val_scores: float, test_scores: float):
    result['dataset'].append(dataset)
    result['task_type'].append(task_type)
    result['model'].append(model)
    result['seed'].append(seed)
    result['features'].append(features)
    result['val_score'].append([round(val_score, 4) for val_score in val_scores])
    result['test_score'].append([round(test_score, 4) for test_score in test_scores])


def process_data(X, y, X_val, y_val, X_test, y_test):
    problem_type = infer_problem_type(y=y)
    label_cleaner = LabelCleaner.construct(problem_type=problem_type, y=y)
    auto_ml_pipeline_feature_generator = AutoMLPipelineFeatureGenerator()
    X = auto_ml_pipeline_feature_generator.fit_transform(X=X)
    X_val = auto_ml_pipeline_feature_generator.transform(X=X_val) if X_val is not None else None
    X_test = auto_ml_pipeline_feature_generator.transform(X=X_test)
    y = label_cleaner.transform(y)
    y_val = label_cleaner.transform(y_val) if y_val is not None else None
    y_test = label_cleaner.transform(y_test)
    return X, y, X_val, y_val, X_test, y_test


for task_type, train_file, test_file in zip(TASK_TYPES, TRAIN_NAMES, TEST_NAMES):
    fit_data = pd.read_csv(os.path.join(DATA_DIR, train_file))
    test_data = pd.read_csv(os.path.join(DATA_DIR, test_file))
    X_all, y_all = fit_data.drop(columns=[args.label]), fit_data[args.label]
    X_test, y_test = test_data.drop(columns=[args.label]), test_data[args.label]
    noise_prefix = 'AG_normal_noise'

    for seed in range(SEEDS):
        print(f"Evaluating {DATA_NAME} {task_type} seed {seed}...")
        for model_class in MODELS:
            result = {'dataset': [], 'task_type': [], 'model': [], 'seed': [], 'features': [], 'val_score': [], 'test_score': []}
            model = model_class()
            X, X_val, y, y_val = train_test_split(X_all, y_all, test_size=int(0.2*len(fit_data)), random_state=seed)
            X, y, X_val, y_val, X_test_new, y_test_new = process_data(X, y, X_val, y_val, X_test, y_test)
            if args.prune_threshold is None:
                X_val_noised = add_noise_column(X_val, noise_prefix)
                X_test_noised = add_noise_column(X_test_new, noise_prefix)
            selector = FeatureSelector(model=model, time_limit=args.time_limit * args.max_fits * 2, keep_models=True)
            selector.select_features(X=X, y=y, X_val=X_val, y_val=y_val, time_limit=args.time_limit,
                                     max_fits=args.max_fits, prune_threshold=args.prune_threshold, stop_threshold=args.stop_threshold)
            trained_models = selector.trained_models
            val_scores, test_scores, all_features = [], [], []
            for trained in trained_models[:len(trained_models) - 1 if args.prune_threshold is None else len(trained_models)]:
                X_val_used = X_val_noised if len([feat for feat in trained.get_features() if noise_prefix in feat]) > 0 else X_val
                val_scores.append(trained.score(X_val_used, y_val))
                X_test_used = X_test_noised if len([feat for feat in trained.get_features() if noise_prefix in feat]) > 0 else X_test_new
                test_scores.append(trained.score(X_test_used, y_test_new))
                all_features.append(trained.features)
            add_datapoint(result, DATA_NAME, task_type, model.name, seed, all_features, val_scores, test_scores)

            bagged_model = BaggedEnsembleModel(model_class(), random_state=seed)
            X_all_new, y_all_new, _, _, X_test_new, y_test_new = process_data(X_all, y_all, None, None, X_test, y_test)
            if args.prune_threshold is None:
                X_test_noised = add_noise_column(X_test_new, noise_prefix)
            selector = FeatureSelector(model=bagged_model, time_limit=args.time_limit * args.max_fits * 2, keep_models=True)
            selector.select_features(X=X_all_new, y=y_all_new, X_val=None, y_val=None, time_limit=args.time_limit,
                                     max_fits=args.max_fits, prune_threshold=args.prune_threshold, stop_threshold=args.stop_threshold)
            trained_models = selector.trained_models
            val_scores, test_scores, all_features = [], [], []
            for trained in trained_models[:len(trained_models) - 1 if args.prune_threshold is None else len(trained_models)]:
                val_scores.append(trained.score_with_oof(y_all_new))
                X_test_used = X_test_noised if len([feat for feat in trained.get_features() if noise_prefix in feat]) > 0 else X_test_new
                test_scores.append(trained.score(X_test_used, y_test_new))
                all_features.append(trained.features)
            add_datapoint(result, DATA_NAME, task_type, f"Bagged{model.name}", seed, all_features, val_scores, test_scores)

            result_df = pd.DataFrame(result)
            if os.path.exists(RESULT_PATH):
                original_result_df = pd.read_csv(RESULT_PATH)
                result_df = pd.concat([original_result_df, result_df], axis=0)
            result_df.to_csv(RESULT_PATH, index=False)
