from autogluon.core import models
from autogluon.core.data import LabelCleaner
from autogluon.core.utils import infer_problem_type, UniformFeatureSelector, NormalFeatureSelector
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
Assess adaptive feature importance computation methods' to accurately
classify whether a feature is relevant or not on selected model and dataset. Split
training dataset into args.seed different training/validation folds and do the following
for each split.
1. Fit a RandomForest model on training data.
2. Call compute_feature_importance() on entire validation set with 1000 shuffles per
   feature as ground truth feature importance. Associate each feature with whether
   it relevant or not.
3. For different resource usage (number of shuffles involving 1000 validation datapoints)
    3a. Call naive_feature_prune() and examine mean parameter per feature. Compare its
        feature relevancy prediction vs ground truth.
    3b. Call bayes_feature_prune() and examine mean parameter per feature. Compare its
        feature relevancy prediction vs ground truth.
4. Summarize score and time elapsed across seeds and place it to args.name/score.csv.
"""


def accuracy(true_param_dict: dict, pred_param_dict: dict, threshold: float):
    # Return fraction of features in true_param_dict and pred_param_dict that lies on the same side of threshold
    num_equal = 0.
    for feature in true_param_dict.keys():
        true_param, pred_param = true_param_dict[feature], pred_param_dict[feature]
        if (true_param['mu'] <= threshold and pred_param['mu'] <= threshold) \
           or (true_param['mu'] > threshold and pred_param['mu'] > threshold):
            num_equal += 1
    return num_equal/len(true_param_dict)


parser = argparse.ArgumentParser()
parser.add_argument('-n', '--name', help='path to save results', type=str, default='fi/UNNAMED')
parser.add_argument('-d', '--dataset', help='path to dataset CSV', type=str, default=None)
parser.add_argument('-l', '--label', help='label column name', type=str, default='class')
parser.add_argument('-s', '--seeds', help='number of seeds to use', type=int, default=1)
parser.add_argument('-r', '--resources', help='a list of number of shuffles to use for each strategy', nargs='+', default=[100])
parser.add_argument('-t', '--true_shuffles', help='number of shuffles to use per feature to compute ground truth', type=int, default=1000)
args = parser.parse_args()
os.makedirs(args.name, exist_ok=True)
RESULT_DIR = args.name
SHUFFLES = args.resources  # number of shuffles of 1000 datapoints that can be allocated among all features

# Load Data
if args.dataset is None:
    fit_data = TabularDataset('https://autogluon.s3.amazonaws.com/datasets/Inc/train.csv')
else:
    fit_data = pd.read_csv(args.dataset)

# On multiple seeds, fit model and evaluate feature importance accuracy
# fit_data = fit_data.head(10000)  # subsample for faster demo
X_all, y_all = fit_data.drop(columns=[args.label]), fit_data[args.label]
naive_accuracies, bayes_accuracies = {}, {}
naive_times, bayes_times = {}, {}
for shuffle in SHUFFLES:
    naive_accuracies[shuffle], bayes_accuracies[shuffle] = [], []
    naive_times[shuffle], bayes_times[shuffle] = [], []

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
    MODEL_PATH = f"{RESULT_DIR}/model_seed{seed}.pkl"
    TRUTH_PATH = f"{RESULT_DIR}/truth_seed{seed}.csv"
    threshold = 0.
    if os.path.exists(MODEL_PATH) and os.path.exists(TRUTH_PATH):
        with open(MODEL_PATH, 'rb') as fp:
            model = pkl.load(fp)
        importance_df = pd.read_csv(TRUTH_PATH, index_col=0)
    else:
        model = RFModel()
        model.fit(X=X, y=y_clean, X_val=X_val, y_val=y_val, random_state=seed)
        with open(MODEL_PATH, 'wb') as fp:
            pkl.dump(model, fp)
        importance_df = model.compute_feature_importance(X_val, y_val_clean, num_shuffle_sets=args.true_shuffles, subsample_size=len(X_val))
        importance_df.to_csv(TRUTH_PATH)
    true_features, true_param_dict = [], {}
    for feature, info in importance_df.iterrows():
        true_param_dict[feature] = {'mu': info['importance']}

    for shuffle in SHUFFLES:
        importance_fn_args = {'X': X_val, 'y': y_val_clean, 'num_shuffle_sets': 1, 'silent': True, 'subsample_size': 1000}
        time_start = time.time()
        uniform_selector = UniformFeatureSelector(importance_fn=model.compute_feature_importance,
                                                  importance_fn_args=importance_fn_args, features=list(X.columns))
        naive_param_dict = uniform_selector.compute_feature_importance(num_resource=shuffle)
        naive_time = time.time() - time_start
        naive_times[shuffle].append(naive_time)
        time_start = time.time()
        bayes_selector = NormalFeatureSelector(importance_fn=model.compute_feature_importance,
                                               importance_fn_args=importance_fn_args, features=list(X.columns))
        bayes_param_dict = bayes_selector.compute_feature_importance(num_resource=shuffle)
        bayes_time = time.time() - time_start
        bayes_times[shuffle].append(bayes_time)
        naive_accuracy = accuracy(true_param_dict, naive_param_dict, threshold)
        bayes_accuracy = accuracy(true_param_dict, bayes_param_dict, threshold)
        print(f"Seed {seed} Shuffle {shuffle} Naive Accuracy: {round(naive_accuracy,5)}, Bayes Accuracy: {round(bayes_accuracy,5)}")
        naive_accuracies[shuffle].append(naive_accuracy)
        bayes_accuracies[shuffle].append(bayes_accuracy)

        result = {"feature": [], "truth": [], "naive": [], "bayes": [], "naive_correct": [], "bayes_correct": []}
        for feature in true_param_dict.keys():
            true_mean = true_param_dict[feature]["mu"]
            naive_mean = naive_param_dict[feature]["mu"]
            bayes_mean = bayes_param_dict[feature]["mu"]
            naive_correct = (true_mean > threshold and naive_mean > threshold) or (true_mean <= threshold and naive_mean <= threshold)
            bayes_correct = (true_mean > threshold and bayes_mean > threshold) or (true_mean <= threshold and bayes_mean <= threshold)
            result["feature"].append(feature)
            result["truth"].append(round(true_mean, 5))
            result["naive"].append(round(naive_mean, 5))
            result["bayes"].append(round(bayes_mean, 5))
            result["naive_correct"].append(naive_correct)
            result["bayes_correct"].append(bayes_correct)
        result_df = pd.DataFrame(result).sort_values(by=["truth"])
        save_path = f"{RESULT_DIR}/shuff{shuffle}_seed{seed}.csv"
        result_df.to_csv(save_path, index=False)

# collect mean accuracy and time taken per shuffle
score_save_path = f"{RESULT_DIR}/score.csv"
score_result = {"method": ["naive", "bayes"]}
for shuffle in SHUFFLES:
    score_result[f"accuracy_shuff{shuffle}"] = [round(sum(naive_accuracies[shuffle]) / args.seeds, 5), round(sum(bayes_accuracies[shuffle]) / args.seeds, 5)]
for shuffle in SHUFFLES:
    score_result[f"time_shuff{shuffle}"] = [round(sum(naive_times[shuffle]) / args.seeds, 5), round(sum(bayes_times[shuffle]) / args.seeds, 5)]
score_result_df = pd.DataFrame(score_result)
score_result_df.to_csv(score_save_path, index=False)
