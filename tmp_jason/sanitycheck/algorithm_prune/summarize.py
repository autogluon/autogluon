import ast
from autogluon.core import data
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

df = pd.read_csv('sanitycheck/algorithm_prune/result.csv')
datasets = df['dataset'].unique()
models = df['model'].unique()
task_types = df['task_type'].unique()
savedir = 'sanitycheck/algorithm_prune/figures'
os.makedirs(savedir, exist_ok=True)

# plot evolution of validation score, test score, and feature types per dataset
for model in models:
    for dataset in datasets:
        for task_type in task_types:
            # collect result for specific model, dataset, and task type config across seeds
            rows = df[(df['dataset'] == dataset) & (df['model'] == model) & (df['task_type'] == task_type)]
            best_val_trajectories, best_val_test_trajectories, best_val_num_original_trajectories, best_val_num_noised_trajectories = [], [], [], []
            for _, row in rows.iterrows():
                feature_trajectory = ast.literal_eval(row.features)
                val_trajectory = ast.literal_eval(row.val_score)
                test_trajectory = ast.literal_eval(row.test_score)
                best_val_trajectory, best_val_test_trajectory, best_val_num_original_trajectory, best_val_num_noised_trajectory = [], [], [], []
                best_val, best_val_test, best_val_num_original, best_val_num_noised = -float('inf'), -float('inf'), -float('inf'), -float('inf')
                for val, test, features in zip(val_trajectory, test_trajectory, feature_trajectory):
                    if val > best_val:
                        best_val = val
                        best_val_test = test
                        best_val_num_original = len([feature for feature in features if 'noise_' != feature[:6]])
                        best_val_num_noised = len([feature for feature in features if 'noise_' == feature[:6]])
                    best_val_trajectory.append(best_val)
                    best_val_test_trajectory.append(best_val_test)
                    best_val_num_original_trajectory.append(best_val_num_original)
                    best_val_num_noised_trajectory.append(best_val_num_noised)
                best_val_trajectories.append(best_val_trajectory)
                best_val_test_trajectories.append(best_val_test_trajectory)
                best_val_num_original_trajectories.append(best_val_num_original_trajectory)
                best_val_num_noised_trajectories.append(best_val_num_noised_trajectory)
            # pad trajectories so they are all of equal length
            max_trajectory_len = 11  # max(list(map(lambda trajectory: len(trajectory), best_val_trajectories)))
            for i in range(len(best_val_trajectories)):
                best_val_trajectory, best_val_test_trajectory = best_val_trajectories[i], best_val_test_trajectories[i]
                best_val_num_original_trajectory, best_val_num_noised_trajectory = best_val_num_original_trajectories[i], best_val_num_noised_trajectories[i]
                fill_len = max_trajectory_len - len(best_val_trajectory)
                best_val_trajectories[i] = best_val_trajectory + fill_len*[best_val_trajectory[-1]]
                best_val_test_trajectories[i] = best_val_test_trajectory + fill_len*[best_val_test_trajectory[-1]]
                best_val_num_original_trajectories[i] = best_val_num_original_trajectory + fill_len*[best_val_num_original_trajectory[-1]]
                best_val_num_noised_trajectories[i] = best_val_num_noised_trajectory + fill_len*[best_val_num_noised_trajectory[-1]]
            # generate plots
            if len(best_val_trajectories) == 0:
                continue
            result_val = np.asarray(best_val_trajectories)
            result_test = np.asarray(best_val_test_trajectories)
            mean_val = result_val.mean(axis=0)
            mean_test = result_test.mean(axis=0)
            std_val = 1.96 * np.std(result_val, axis=0)
            std_test = 1.96 * np.std(result_test, axis=0)
            result_num_original = np.asarray(best_val_num_original_trajectories)
            result_num_noised = np.asarray(best_val_num_noised_trajectories)
            mean_num_original = result_num_original.mean(axis=0)
            mean_num_noised = result_num_noised.mean(axis=0)
            std_num_original = np.std(result_num_original, axis=0)
            std_num_noised = np.std(result_num_noised, axis=0)
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
            ax[1].plot(x, mean_num_original, color='g')
            ax[1].fill_between(x, mean_num_original-std_num_original, mean_num_original+std_num_original, color='g', alpha=.1)
            ax[1].set_title("Number of Kept Features")
            ax[1].set_xlabel("Number of Model Fits")
            ax[1].set_ylabel("Number of Kept Features")
            ax[1].plot(x, mean_num_noised, color='y')
            ax[1].fill_between(x, mean_num_noised-std_num_noised, mean_num_noised+std_num_noised, color='y', alpha=.1)
            ax[1].legend([f"# Original Features", f"# Synthetic Features"])
            fig.suptitle(f"{model} Stats on Dataset {dataset} of Type {task_type}")
            fig.tight_layout()
            fig.savefig(f'{savedir}/{dataset}_{task_type}_{model}.png')
