"""
Show that perfect feature pruning empirically improves test set accuracy on 5 datasets
for LightGBM and NNFastAiTabularModel. Create synthetic datasets involving white noise
and feature permutation columns 1.2x and 2x the size of original datasets. Compare model
accuracies before and after adding synthetic features. Try both bagged and non bagged
versions. Datasets are adult, airlines, australian, covertype, and higgs. In addition,
show that permutation feature importance is an indicative measure of whether a feature
is from the original dataset or not.
"""

import argparse
import os
from numpy.core.defchararray import index
import pandas as pd
from autogluon.core.models import BaggedEnsembleModel
from autogluon.tabular.models import LGBModel, NNFastAiTabularModel
from autogluon.core.data import LabelCleaner
from autogluon.core.utils import infer_problem_type
from autogluon.features.generators import AutoMLPipelineFeatureGenerator
from sklearn.model_selection import train_test_split


parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data_dir', help='dataset directory', type=str, required=True)
parser.add_argument('-l', '--label', help='label column name', type=str, default='class')
parser.add_argument('-r', '--result_path', help='file to save test set score to', type=str, default='sanitycheck/result.csv')
parser.add_argument('-t', '--time_limit', help='time limit models have to train in seconds', type=int, default=3600)
args = parser.parse_args()

MODELS = [LGBModel, NNFastAiTabularModel]
DATA_DIR = args.data_dir
RESULT_PATH = args.result_path
SEED = 0
DATA_NAME = os.path.basename(DATA_DIR)
TASK_TYPES = ['original', 'normal_easy', 'normal_hard', 'shuffle_easy', 'shuffle_hard']
TRAIN_NAMES = ['train_data.csv', 'n1_r0.2_train_data.csv', 'n1_r1.0_train_data.csv', 'n2_r0.2_train_data.csv', 'n2_r1.0_train_data.csv']
TEST_NAMES = ['test_data.csv', 'n1_r0.2_test_data.csv', 'n1_r1.0_test_data.csv', 'n2_r0.2_test_data.csv', 'n2_r1.0_test_data.csv']


for task_type, train_file, test_file in zip(TASK_TYPES, TRAIN_NAMES, TEST_NAMES):
    fit_data = pd.read_csv(os.path.join(DATA_DIR, train_file))
    test_data = pd.read_csv(os.path.join(DATA_DIR, test_file))
    X_all, y_all = fit_data.drop(columns=[args.label]), fit_data[args.label]
    X_test, y_test = test_data.drop(columns=[args.label]), test_data[args.label]

    print(f"Evaluating {DATA_NAME} {task_type}...")
    for model_class in MODELS:
        result = {'dataset': [], 'task_type': [], 'model': [], 'seed': [], 'features': [], 'fi_mean': [], 'fi_pval': [], 'val_score': [], 'test_score': []}
        model = model_class()
        X, X_val, y, y_val = train_test_split(X_all, y_all, test_size=int(0.2*len(fit_data)), random_state=SEED)
        problem_type = infer_problem_type(y=y)
        label_cleaner = LabelCleaner.construct(problem_type=problem_type, y=y)
        auto_ml_pipeline_feature_generator = AutoMLPipelineFeatureGenerator()
        X = auto_ml_pipeline_feature_generator.fit_transform(X=X)
        y = label_cleaner.transform(y)
        X_val = auto_ml_pipeline_feature_generator.transform(X=X_val)
        y_val = label_cleaner.transform(y_val)
        model = model.fit(X=X, y=y, X_val=X_val, y_val=y_val, time_limit=args.time_limit)
        X_test_new = auto_ml_pipeline_feature_generator.transform(X=X_test)
        y_test_new = label_cleaner.transform(y_test)
        model_val_score = model.score(X_val, y_val)
        model_test_score = model.score(X_test_new, y_test_new)
        feature_importance = model.compute_feature_importance(X=X_val, y=y_val, num_shuffle_sets=10, subsample_size=5000)
        features = feature_importance.index.tolist()
        fis = [round(fi, 4) for fi in feature_importance['importance'].tolist()]
        pvals = [round(pval, 4) for pval in feature_importance['p_value'].tolist()]
        result['dataset'].append(DATA_NAME)
        result['task_type'].append(task_type)
        result['model'].append(model.name)
        result['seed'].append(0)
        result['features'].append(features)
        result['fi_mean'].append(fis)
        result['fi_pval'].append(pvals)
        result['val_score'].append(round(model_val_score, 4))
        result['test_score'].append(round(model_test_score, 4))

        bagged_model = BaggedEnsembleModel(model_class(), random_state=SEED)
        problem_type = infer_problem_type(y=y_all)
        label_cleaner = LabelCleaner.construct(problem_type=problem_type, y=y_all)
        auto_ml_pipeline_feature_generator = AutoMLPipelineFeatureGenerator()
        X_all_new = auto_ml_pipeline_feature_generator.fit_transform(X=X_all)
        y_all_new = label_cleaner.transform(y_all)
        base_model_name = bagged_model.model_base.name
        bagged_model = bagged_model.fit(X=X_all_new, y=y_all_new, time_limit=args.time_limit)
        X_test_new = auto_ml_pipeline_feature_generator.transform(X=X_test)
        y_test_new = label_cleaner.transform(y_test)
        bagged_model_val_score = bagged_model.score_with_oof(y_all_new)
        bagged_model_test_score = bagged_model.score(X_test_new, y_test_new)
        bagged_feature_importance = model.compute_feature_importance(X=X, y=y, num_shuffle_sets=10, subsample_size=5000, is_oof=True)
        features = bagged_feature_importance.index.tolist()
        bagged_fis = [round(fi, 4) for fi in bagged_feature_importance['importance'].tolist()]
        bagged_pvals = [round(pval, 4) for pval in bagged_feature_importance['p_value'].tolist()]
        result['dataset'].append(DATA_NAME)
        result['task_type'].append(task_type)
        result['model'].append(f"Bagged{base_model_name}")
        result['seed'].append(0)
        result['features'].append(features)
        result['fi_mean'].append(bagged_fis)
        result['fi_pval'].append(bagged_pvals)
        result['val_score'].append(round(bagged_model_val_score, 4))
        result['test_score'].append(round(bagged_model_test_score, 4))

        result_df = pd.DataFrame(result)
        if os.path.exists(RESULT_PATH):
            original_result_df = pd.read_csv(RESULT_PATH)
            result_df = pd.concat([original_result_df, result_df], axis=0)
        result_df.to_csv(RESULT_PATH, index=False)
