"""
Plot the evolution of validation/test set scores on fit_with_prune model refits.
1. For each seed,
    1a. Initialize model and run fit_with_prune to obtain a list of models.
    1b. On each model, run score to obtain its test set performance.
2. For each fit, compute average validation and test set scores.
3. Plot two sequences.
"""

from autogluon.core.data import LabelCleaner
from autogluon.core.utils import infer_problem_type
from autogluon.core.models import BaggedEnsembleModel
from autogluon.tabular.models.rf.rf_model import RFModel
from autogluon.tabular.models import KNNModel, RFModel, CatBoostModel, NNFastAiTabularModel, LGBModel
from autogluon.tabular import TabularDataset
from autogluon.features.generators import AutoMLPipelineFeatureGenerator
from sklearn.model_selection import train_test_split
import argparse
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument('-n', '--name', help='directory to save results', type=str, default='plots/score/UNNAMED')
parser.add_argument('-f', '--train_path', help='path to train dataset CSV', type=str, default=None)
parser.add_argument('-g', '--test_path', help='path to test dataset CSV', type=str, default=None)
parser.add_argument('-l', '--label', help='label column name', type=str, default='class')
parser.add_argument('-s', '--seeds', help='number of seeds to use', type=int, default=2)
parser.add_argument('-m', '--max_num_fit', help='maximum number of times the model will be fitted', type=int, default=50)
parser.add_argument('-d', '--stop_threshold', help='if score does not improve for this many iterations, stop feature pruning', type=int, default=3)
parser.add_argument('-p', '--prune_ratio', help='prune at most this amount of features at once per pruning iteration', type=float, default=0.05)
parser.add_argument('-r', '--resource', help='number of shuffles to evaluate per model fit iteration', type=int, default=None)
parser.add_argument('-t', '--strategy', help='which strategy to evaluate', type=str, default='uniform', choices=['uniform', 'backwardsearch'])
parser.add_argument('-u', '--subsample_size', help='how many subsamples to use per shuffle', type=int, default=5000)
parser.add_argument('-z', '--mode', help='which model to use', type=str, default='catboost', choices=['randomforest', 'catboost', 'fastai', 'knn', 'lightgbm'])
parser.add_argument('-b', '--bagged', help='whether to bag models. 0 for false and 1 for true.', type=int, default=0, choices=[0, 1])


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

# import pdb; pdb.set_trace() ## NOTE: Try fitting with only noised data
# fit_data = fit_data[[feature for feature in fit_data.columns if 'noise_' in feature or args.label == feature]]
# test_data = test_data[[feature for feature in test_data.columns if 'noise_' in feature or args.label == feature]]

# On multiple seeds, fit model and evaluate accuracy
# fit_data = fit_data.head(10000)  # subsample for faster demo
X_all, y_all = fit_data.drop(columns=[args.label]), fit_data[args.label]
X_test, y_test = test_data.drop(columns=[args.label]), test_data[args.label]
val_trajectories, test_trajectories = [], []
num_original_features, num_noised_features = [], []
for seed in range(args.seeds):
    auto_ml_pipeline_feature_generator = AutoMLPipelineFeatureGenerator()
    if args.mode == 'autogluon':
        # TODO: Enable this with full autogluon run
        pass
    else:
        # call fit_with_prune, return all models, and do stuff there
        if args.mode == 'randomforest':
            model = RFModel()
        elif args.mode == 'fastai':
            model = NNFastAiTabularModel()
        elif args.mode == 'knn':
            model = KNNModel()
        elif args.mode == 'lightgbm':
            model = LGBModel()
        else:
            model = CatBoostModel()
        if args.strategy == 'uniform':
            fi_strategy = 'uniform'
            fp_strategy = 'percentage'
        else:
            fi_strategy = 'backwardsearch'
            fp_strategy = 'single'

        # clean data and call fit_with_prune
        if args.bagged:
            model = BaggedEnsembleModel(model, random_state=seed)
            problem_type = infer_problem_type(y=y_all)
            label_cleaner = LabelCleaner.construct(problem_type=problem_type, y=y_all)
            X_all_new = auto_ml_pipeline_feature_generator.fit_transform(X=X_all)
            y_all_new = label_cleaner.transform(y_all)
            best_model, all_model_info = model.fit_with_prune(X=X_all_new, y=y_all_new, X_val=None, y_val=None, max_num_fit=args.max_num_fit,
                                                              stop_threshold=args.stop_threshold, prune_ratio=0.1, num_resource=args.resource,
                                                              fi_strategy=fi_strategy, fp_strategy=fp_strategy, subsample_size=args.subsample_size,
                                                              prune_threshold=0.001)
        else:
            X, X_val, y, y_val = train_test_split(X_all, y_all, test_size=int(0.2*len(fit_data)), random_state=seed)
            problem_type = infer_problem_type(y=y)
            label_cleaner = LabelCleaner.construct(problem_type=problem_type, y=y)
            X = auto_ml_pipeline_feature_generator.fit_transform(X=X)
            y = label_cleaner.transform(y)
            X_val = auto_ml_pipeline_feature_generator.transform(X=X_val)
            y_val = label_cleaner.transform(y_val)
            best_model, all_model_info = model.fit_with_prune(X=X, y=y, X_val=X_val, y_val=y_val, max_num_fit=args.max_num_fit,
                                                              stop_threshold=args.stop_threshold, prune_ratio=0.1, num_resource=args.resource,
                                                              fi_strategy=fi_strategy, fp_strategy=fp_strategy, subsample_size=args.subsample_size, 
                                                              prune_threshold=0.001)
        X_test_new = auto_ml_pipeline_feature_generator.transform(X=X_test)
        y_test_new = label_cleaner.transform(y_test)

        val_trajectory, test_trajectory = [], []
        num_original_feature, num_noised_feature = [], []
        best_val_score, best_val_score_test = -float('inf'), -float('inf')
        # plot evolution of test score of model with best validation loss
        for i, model_info in enumerate(all_model_info):
            val_score, model = model_info
            if val_score > best_val_score:
                best_val_score = val_score
                best_val_score_test = model.score(X_test_new, y_test_new)
                best_num_original_feature = len([feature for feature in model.features if 'noise_' not in feature])
                best_num_noised_feature = len([feature for feature in model.features if 'noise_' in feature])
            val_trajectory.append(best_val_score)
            test_trajectory.append(best_val_score_test)
            num_original_feature.append(best_num_original_feature)
            num_noised_feature.append(best_num_noised_feature)
        val_trajectories.append(val_trajectory)
        test_trajectories.append(test_trajectory)
        num_original_features.append(num_original_feature)
        num_noised_features.append(num_noised_feature)

# pad trajectories so they are all of equal length
max_trajectory_len = max(list(map(lambda trajectory: len(trajectory), val_trajectories)))
for i in range(len(val_trajectories)):
    val_trajectory, test_trajectory = val_trajectories[i], test_trajectories[i]
    fill_len = max_trajectory_len - len(val_trajectory)
    val_trajectories[i] = val_trajectory + fill_len*[val_trajectory[-1]]
    test_trajectories[i] = test_trajectory + fill_len*[test_trajectory[-1]]
    num_original_feature, num_noised_feature = num_original_features[i], num_noised_features[i]
    num_original_features[i] = num_original_feature + fill_len*[num_original_feature[-1]]
    num_noised_features[i] = num_noised_feature + fill_len*[num_noised_feature[-1]]

# plot
result_val = np.asarray(val_trajectories)
result_test = np.asarray(test_trajectories)
mean_val = result_val.mean(axis=0)
mean_test = result_test.mean(axis=0)
std_val = 1.96 * np.std(result_val, axis=0)
std_test = 1.96 * np.std(result_test, axis=0)
result_orig_feat = np.asarray(num_original_features)
result_noised_feat = np.asarray(num_noised_features)
mean_orig_feat = result_orig_feat.mean(axis=0)
mean_noised_feat = result_noised_feat.mean(axis=0)
std_orig_feat = np.std(result_orig_feat, axis=0)
std_noised_feat = np.std(result_noised_feat, axis=0)
x = [i+1 for i in range(max_trajectory_len)]
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].plot(x, mean_val, color='r')
ax[0].fill_between(x, mean_val-std_val, mean_val+std_val, color='r', alpha=.1)
ax[0].set_title(f"Validation and Test Set Scores")
ax[0].set_xlabel("Number of Model Fits")
ax[0].set_ylabel("Accuracy")
ax[0].plot(x, mean_test, color='b')
ax[0].fill_between(x, mean_test-std_test, mean_test+std_test, color='b', alpha=.1)
ax[0].legend([f"Val ({round(mean_val[0],4)}=>{round(mean_val[-1],4)})", f"Test ({round(mean_test[0],4)}=>{round(mean_test[-1],4)})"])
ax[1].plot(x, mean_orig_feat, color='g')
ax[1].fill_between(x, mean_orig_feat-std_orig_feat, mean_orig_feat+std_orig_feat, color='g', alpha=.1)
ax[1].set_title("Number of Kept Features")
ax[1].set_xlabel("Number of Model Fits")
ax[1].set_ylabel("Number of Kept Features")
ax[1].plot(x, mean_noised_feat, color='y')
ax[1].fill_between(x, mean_noised_feat-std_noised_feat, mean_noised_feat+std_noised_feat, color='y', alpha=.1)
ax[1].legend([f"# Original Features", f"# Synthetic Features"])
fig.suptitle(f"{'Bagged ' if args.bagged else ''}{args.mode.upper()} Stats From Strategy: {args.strategy}")
fig.tight_layout()
fig.savefig(f'{RESULT_DIR}/evolution.png')

# save trajectories
result_val_df = pd.DataFrame(val_trajectories)
result_test_df = pd.DataFrame(test_trajectories)
result_val_df.to_csv(f'{RESULT_DIR}/val.csv', index=False)
result_test_df.to_csv(f'{RESULT_DIR}/test.csv', index=False)
