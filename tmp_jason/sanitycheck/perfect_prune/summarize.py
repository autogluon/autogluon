import ast
import pandas as pd
from sklearn.metrics import f1_score

df = pd.read_csv('sanitycheck/perfect_prune/result.csv')
datasets = df['dataset'].unique()
models = df['model'].unique()

# print average rank of noise configuration per model across datasets
mean_rank_rows = []
for model in models:
    mean_val_ranks = None
    mean_test_ranks = None
    for dataset in datasets:
        rows = df[(df['dataset'] == dataset) & (df['model'] == model)]
        val_ranks = rows['val_score'].rank(ascending=False).to_numpy()
        if mean_val_ranks is None:
            mean_val_ranks = val_ranks
        else:
            mean_val_ranks += val_ranks
        test_ranks = rows['test_score'].rank(ascending=False).to_numpy()
        if mean_test_ranks is None:
            mean_test_ranks = test_ranks
        else:
            mean_test_ranks += test_ranks
    mean_val_ranks = mean_val_ranks / len(datasets)
    mean_test_ranks = mean_test_ranks / len(datasets)
    mean_rank_dict = {
        'model': model, 'val_original': mean_val_ranks[0], 'val_normal1': mean_val_ranks[1],
        'val_normal2': mean_val_ranks[2], 'val_shuffle1': mean_val_ranks[3], 'val_shuffle2': mean_val_ranks[4],
        'test_original': mean_test_ranks[0], 'test_normal1': mean_test_ranks[1], 'test_normal2': mean_test_ranks[2],
        'test_shuffle1': mean_test_ranks[3], 'test_shuffle2': mean_test_ranks[4],
    }
    mean_rank_rows.append(pd.Series(mean_rank_dict))

mean_rank_df = pd.DataFrame(mean_rank_rows).set_index('model')
mean_rank_df.to_csv('sanitycheck/perfect_prune/summarized_score.csv')

# summarize f1 score based on feature importance
fi_info = []
for model in models:
    row_count = 0
    f1_means, f1_pvals = [], []
    for dataset in datasets:
        rows = df[(df['dataset'] == dataset) & (df['model'] == model)]
        for _, row in rows.iterrows():
            features = ast.literal_eval(row.features)
            fi_mean = ast.literal_eval(row.fi_mean)
            fi_pval = ast.literal_eval(row.fi_pval)
            features_info = [el for el in zip(features, fi_mean, fi_pval)]
            k = len([feature for feature in features if 'noise_' not in feature])
            top_k_fi_mean = list(map(lambda e: e[0], sorted(features_info, key=lambda e: e[1])[::-1][:k]))
            top_k_fi_pval = list(map(lambda e: e[0], sorted(features_info, key=lambda e: e[2])[:k]))
            # import pdb; pdb.set_trace()
            truth = [1 if 'noise_' not in feature else 0 for feature in features]
            pred_fi_mean = [1 if feature in top_k_fi_mean else 0 for feature in features]
            pred_fi_pval = [1 if feature in top_k_fi_pval else 0 for feature in features]
            f1_means.append(f1_score(truth, pred_fi_mean))
            f1_pvals.append(f1_score(truth, pred_fi_pval))
            row_count += 1
    model_f1_mean = sum(f1_means) / row_count
    model_f1_pval = sum(f1_pvals) / row_count
    fi_info.append(pd.Series({'model': model, 'f1_mean': model_f1_mean, 'f1_pval': model_f1_pval}))

fi_df = pd.DataFrame(fi_info).set_index('model')
fi_df.to_csv('sanitycheck/perfect_prune/summarized_fi.csv')
