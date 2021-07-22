from autogluon.core import models
from autogluon.core.data import LabelCleaner
from autogluon.core.utils import infer_problem_type, UniformFeatureSelector, NormalFeatureSelector
from autogluon.tabular.models.rf.rf_model import RFModel
from autogluon.tabular.models.lgb.lgb_model import LGBModel
from autogluon.tabular import TabularDataset
from autogluon.features.generators import AutoMLPipelineFeatureGenerator
from sklearn.model_selection import train_test_split
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle as pkl
import time

"""
Plot the evolution of TP/FP/FN/TN feature importance statistics.
1. For each seed
    1a. Initialize model and compute ground truth importance scores. Save the results.
    1b. Keep track of changing feature importance estimates in a Pandas dataframe. Each feature
        importance computation should append a row to the dataframe on updated feature importance
        scores for all features.
    1c. Save the results onto disk.
    1c. For every timestep (dataframe rows), compute TP/FP/FN/TN estimates.
2. For each fit, compute average TP/FP/FN/TN statistics.
3. Plot accuracy, precision, recall, and F1 scores.

NOTE: Make sure we can easily add new results. This can be done by plotting stuff read from disk.
Goal: Finish and run this by end of tomorrow.

Directory Structure
- session name
    - method1
        - seed1.csv
        - seed2.csv
    - method2
        - seed1.csv
        - seed2.csv
"""

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--name', help='path to save results', type=str, default='plots/fi/UNNAMED')
parser.add_argument('-d', '--dataset', help='path to dataset CSV', type=str, default=None)
parser.add_argument('-l', '--label', help='label column name', type=str, default='class')
parser.add_argument('-s', '--seeds', help='number of seeds to use', type=int, default=2)
parser.add_argument('-r', '--resource', help='a list of number of shuffles to use for each strategy', type=int, default=28)
parser.add_argument('-t', '--true_shuffles', help='number of shuffles to use per feature to compute ground truth', type=int, default=1000)
args = parser.parse_args()
os.makedirs(args.name, exist_ok=True)
RESULT_DIR = args.name
SHUFFLES = args.resource  # number of shuffles of 1000 datapoints that can be allocated among all features
STRATEGIES = [NormalFeatureSelector]
THRESHOLD = 0.001

# Load Data
if args.dataset is None:
    fit_data = TabularDataset('https://autogluon.s3.amazonaws.com/datasets/Inc/train.csv')
else:
    fit_data = pd.read_csv(args.dataset)

# On multiple seeds, fit model and evaluate feature importance accuracy
# fit_data = fit_data.head(10000)  # subsample for faster demo
X_all, y_all = fit_data.drop(columns=[args.label]), fit_data[args.label]
all_statistics = []
for seed in range(args.seeds):
    X, X_val, y, y_val = train_test_split(X_all, y_all, test_size=int(0.2*len(fit_data)), random_state=seed)
    auto_ml_pipeline_feature_generator = AutoMLPipelineFeatureGenerator()
    X = auto_ml_pipeline_feature_generator.fit_transform(X=X)
    X_val = auto_ml_pipeline_feature_generator.transform(X=X_val)
    problem_type = infer_problem_type(y=y)
    label_cleaner = LabelCleaner.construct(problem_type=problem_type, y=y)
    y_clean = label_cleaner.transform(y)
    y_val_clean = label_cleaner.transform(y_val)

    # Fit model and compute ground truth importance scores
    # If this was done before, simply recycle the results from the previous run
    model_path = f"{RESULT_DIR}/model_seed{seed}.pkl"
    truth_path = f"{RESULT_DIR}/truth_seed{seed}.csv"
    threshold = 0.
    if os.path.exists(model_path) and os.path.exists(truth_path):
        with open(model_path, 'rb') as fp:
            model = pkl.load(fp)
        importance_df = pd.read_csv(truth_path, index_col=0)
    else:
        model = RFModel()
        model.fit(X=X, y=y_clean, X_val=X_val, y_val=y_val, random_state=seed)
        with open(model_path, 'wb') as fp:
            pkl.dump(model, fp)
        importance_df = model.compute_feature_importance(X_val, y_val_clean, num_shuffle_sets=args.true_shuffles, subsample_size=len(X_val))
        importance_df.to_csv(truth_path)

    all_strategy_dfs = {}
    importance_fn_args = {'X': X_val, 'y': y_val_clean, 'num_shuffle_sets': 1, 'silent': True, 'subsample_size': 1000}
    for selector_type in STRATEGIES:
        strategy = selector_type.__name__
        selector_path = f"{RESULT_DIR}/{strategy}_seed{seed}.csv"
        if os.path.exists(selector_path):
            strategy_df = pd.read_csv(selector_path)
        else:
            selector = selector_type(importance_fn=model.compute_feature_importance,
                                     importance_fn_args=importance_fn_args, features=list(X.columns))
            time_start = time.time()
            param_dict, trajectories = selector.compute_feature_importance(num_resource=args.resource)
            time_elapsed = time.time() - time_start
            # trajectories: {<feature>: [<param_dict_1 (mu, sigma, num_pulled, num_timestep)>, <param_dict_2>...]}
            # convert this to DataFrame with row: timestep, col: feature importance mu
            # for iteration in args.resource
            #   add row to dataframe where a row has feature importance mu for all features
            strategy_dfs = []
            for iteration in range(args.resource):
                row = {}
                for feature, trajectory in trajectories.items():
                    param_dict = trajectories[feature][iteration]
                    # mean, latest_pull_iter = param_dict['mu'], param_dict['latest_pull_iter']
                    row[feature] = [param_dict['mu']]
                    # if latest_pull_iter == iteration:
                    #     row[feature] = param_dict['mu']
                    # else:
                    #     row[feature] = trajectories[feature][iteration-1]['mu']
                strategy_dfs.append(pd.DataFrame(row))
            strategy_df = pd.concat(strategy_dfs, ignore_index=True)
            strategy_df.to_csv(selector_path, index=False)
        all_strategy_dfs[strategy] = strategy_df

    # Calculate TP, FP, FN, TN and save them to DataFrame with row: timestep, col: metrics
    all_relevance_dfs = {}
    true_scores = importance_df['importance']
    for strategy, strategy_df in all_strategy_dfs.items():
        relevance_dfs = []
        for iteration in range(args.resource):
            iter_scores = strategy_df.iloc[iteration]
            num_total_pos, num_total_neg, num_correct_pos, num_correct_neg = 0, 0, 0, 0
            for feature in true_scores.keys():
                true_score, iter_score = true_scores[feature], iter_scores[feature]
                if true_score > THRESHOLD:
                    num_total_pos += 1
                    if iter_score > THRESHOLD:
                        num_correct_pos += 1
                else:
                    num_total_neg += 1
                    if iter_score <= THRESHOLD:
                        num_correct_neg += 1
            row = {
                'accuracy': [(num_correct_pos+num_correct_neg)/(num_total_pos+num_total_neg)],
                'f1': [(2*num_correct_pos)/(2*num_correct_pos+(num_total_pos-num_correct_pos)+(num_total_neg-num_correct_neg))],
                'sensitivity': [num_correct_pos/num_total_pos],
                'specificity': [num_correct_neg/num_total_neg],
            }
            relevance_dfs.append(pd.DataFrame(row))
        relevance_df = pd.concat(relevance_dfs, ignore_index=True)
        all_relevance_dfs[strategy] = relevance_df
    all_statistics.append(all_relevance_dfs)

plot_save_path = f"{RESULT_DIR}/evolution.png"
fig, ax = plt.subplots(2, 2, figsize=(8, 8), sharey=True)
selector_names = [selector_type.__name__ for selector_type in STRATEGIES]
for strategy in selector_names:
    mean_accuracies = np.mean([stats[strategy]['accuracy'] for stats in all_statistics], axis=0)
    mean_f1s = np.mean([stats[strategy]['f1'] for stats in all_statistics], axis=0)
    mean_sensitivities = np.mean([stats[strategy]['sensitivity'] for stats in all_statistics], axis=0)
    mean_specificities = np.mean([stats[strategy]['specificity'] for stats in all_statistics], axis=0)
    x = [i+1 for i in range(len(mean_accuracies))]
    ax[0, 0].plot(x, mean_accuracies)
    ax[0, 1].plot(x, mean_f1s)
    ax[1, 0].plot(x, mean_sensitivities)
    ax[1, 1].plot(x, mean_specificities)

ax[0, 0].set_title("Accuracy")
ax[0, 1].set_title("F1 Score")
ax[1, 0].set_title("True Positive Rate")
ax[1, 1].set_title("True Negative Rate")
ax[0, 0].set_xlabel("Number of Resource Allocation")
ax[0, 1].set_xlabel("Number of Resource Allocation")
ax[1, 0].set_xlabel("Number of Resource Allocation")
ax[1, 1].set_xlabel("Number of Resource Allocation")
ax[0, 0].legend(selector_names)
ax[0, 1].legend(selector_names)
ax[1, 0].legend(selector_names)
ax[1, 1].legend(selector_names)

fig.suptitle(f"Feature Relevance Estimation ({args.seeds} seeds)")
fig.tight_layout()
fig.savefig(plot_save_path)
