from autogluon.core import models
from autogluon.core.data import LabelCleaner
from autogluon.core.utils import infer_problem_type
from autogluon.tabular.models.rf.rf_model import RFModel
from autogluon.tabular.models.lgb.lgb_model import LGBModel
from autogluon.tabular import TabularDataset
from autogluon.features.generators import AutoMLPipelineFeatureGenerator
from sklearn.model_selection import train_test_split
import argparse
import os
import pandas as pd
import pickle as pkl
import time

"""
Assess adaptive feature importance computation methods' ability to improve
test set performance. Split training dataset into args.seed different training/
validation fold and do the following for each split.
1. Call fit() on a RandomForest model and score it on test set
2. For each feature importance computation method
    2a. Call fit_with_prune() on a RandomForest model with that method
    2b. Score the best model on test set
3. Summarize score and time elapsed across seeds and place it to args.name/score.csv
"""

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--name', help='path to save results', type=str, default='score/UNNAMED')
parser.add_argument('-f', '--train_path', help='path to train dataset CSV', type=str, default=None)
parser.add_argument('-g', '--test_path', help='path to test dataset CSV', type=str, default=None)
parser.add_argument('-l', '--label', help='label column name', type=str, default='class')
parser.add_argument('-s', '--seeds', help='number of seeds to use', type=int, default=1)
args = parser.parse_args()
os.makedirs(args.name, exist_ok=True)
RESULT_DIR = args.name

# Load Data
if args.train_path is None:
    fit_data = TabularDataset('https://autogluon.s3.amazonaws.com/datasets/Inc/train.csv')
else:
    fit_data = pd.read_csv(args.train_path)
if args.test_path is None:
    test_data = TabularDataset('https://autogluon.s3.amazonaws.com/datasets/Inc/test.csv')
else:
    test_data = pd.read_csv(args.test_path)

# On multiple seeds, fit model and evaluate accuracy
# fit_data = fit_data.head(10000)  # subsample for faster demo
X_all, y_all = fit_data.drop(columns=[args.label]), fit_data[args.label]
X_test, y_test = test_data.drop(columns=[args.label]), test_data[args.label]
accuracies = {'method': ['keepall', 'naive', 'bayes']}
for seed in range(args.seeds):
    # clean data
    X, X_val, y, y_val = train_test_split(X_all, y_all, test_size=int(0.2*len(fit_data)), random_state=seed)
    auto_ml_pipeline_feature_generator = AutoMLPipelineFeatureGenerator()
    X = auto_ml_pipeline_feature_generator.fit_transform(X=X)
    X_val = auto_ml_pipeline_feature_generator.transform(X=X_val)
    X_test_new = auto_ml_pipeline_feature_generator.transform(X=X_test)
    problem_type = infer_problem_type(y=y)
    label_cleaner = LabelCleaner.construct(problem_type=problem_type, y=y)
    y = label_cleaner.transform(y)
    y_val = label_cleaner.transform(y_val)
    y_test_new = label_cleaner.transform(y_test)

    # evaluate base model accuracy
    model = RFModel()
    model.fit(X=X, y=y, X_val=X_val, y_val=y_val, random_state=seed)
    base_score = model.score(X_test_new, y_test_new)
    # evaluate naive pruning accuracy
    model = RFModel()
    model = model.fit_with_prune(X=X, y=y, X_val=X_val, y_val=y_val, stop_threshold=3, strategy='naive')
    naive_score = model.score(X_test_new, y_test_new)
    # evaluate bayes pruning accuracy
    model = RFModel()
    model = model.fit_with_prune(X=X, y=y, X_val=X_val, y_val=y_val, stop_threshold=3, strategy='bayes')
    bayes_score = model.score(X_test_new, y_test_new)

    accuracies[f"accuracy_seed{seed}"] = [base_score, naive_score, bayes_score]

# Compute mean accuracy per method and add to dataframe
accuracy_df = pd.DataFrame(accuracies)
accuracy_df["accuracy_mean"] = accuracy_df.mean(axis=1)
accuracy_save_path = f"{RESULT_DIR}/score.csv"
accuracy_df.to_csv(accuracy_save_path, index=False)
