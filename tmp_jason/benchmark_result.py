import pandas as pd


def add_dataset_info(results_raw: pd.DataFrame, task_metadata: pd.DataFrame):
    results_raw['tid'] = [int(x.split('/')[-1]) for x in results_raw['id']]
    task_metadata['ClassRatio'] = task_metadata['MinorityClassSize'] / task_metadata['NumberOfInstances']
    results_raw = results_raw.merge(task_metadata, on=['tid'])
    return results_raw


def compare_dfs(df1, df2, metric):
    df1_better, equal_performance, df2_better = [], [], []
    for _, row in df1.iterrows():
        task, df1_score = row["task"], row[metric]
        df2_score = df2[df2["task"] == task][metric].item()
        if df1_score > df2_score:
            df1_better.append(task)
        elif df1_score < df2_score:
            df2_better.append(task)
        else:
            equal_performance.append(task)
    return df1_better, equal_performance, df2_better

"""
base = pd.read_csv("result/results_automlbenchmark_1h8c_autogluon.ag.1h8c.aws.20210629T001407.csv")
uniform = pd.read_csv("result/results_automlbenchmark_4h8c_autogluon_prune_uniform.ag.4h8c.aws.20210722T055042.csv")
backward = pd.read_csv("result/results_automlbenchmark_4h8c_autogluon_prune_backwardsearch.ag.4h8c.aws.20210722T055046.csv")
"""

# base = pd.read_csv("~/Downloads/results_automlbenchmark_12h8c_autogluon_bestquality_norepeat.ag.12h8c.aws.20210723T212945.csv")
# uniform = pd.read_csv("~/Downloads/results_automlbenchmark_12h8c_autogluon_prune_uniform_bestquality.ag.12h8c.aws.20210723T213246.csv")
# base = pd.read_csv("~/Downloads/results_automlbenchmark_12h8c_autogluon_bestquality.ag.12h8c.aws.20210725T042534.csv")
base = pd.read_csv("~/Downloads/results_automlbenchmark_12h8c_autogluon_prune_backwardsearch_bestquality.ag.12h8c.aws.20210725T042219.csv")
uniform = pd.read_csv("~/Downloads/results_automlbenchmark_12h8c_autogluon_prune_uniform_bestquality.ag.12h8c.aws.20210725T042156.csv")
task_metadata = pd.read_csv('result/task_metadata.csv')
base = add_dataset_info(base, task_metadata)
uniform = add_dataset_info(uniform, task_metadata)
# backward = add_dataset_info(backward, task_metadata)

basebin = base[base["type"] == "binary"]
uniformbin = uniform[uniform["type"] == "binary"]
# backwardbin = backward[backward["type"] == "binary"]
basecat = base[base["type"] == "multiclass"]
uniformcat = uniform[uniform["type"] == "multiclass"]
# backwardcat = backward[backward["type"] == "multiclass"]
basereg = base[base["type"] == "regression"]
uniformreg = uniform[uniform["type"] == "regression"]
# backwardreg = backward[backward["type"] == "regression"]

import pdb; pdb.set_trace()
"""
first_better, equal_performance, second_better = compare_dfs(base[(50000 > base["NumberOfInstances"]) & (base["NumberOfInstances"] > 0)],
                                                             backward[(50000 > backward["NumberOfInstances"]) & (backward["NumberOfInstances"] > 0)],
                                                             "result")
print("=== Large Dataset (+5000 samples) Results ===")
print(f"Base Win: {len(first_better)}, Search Win: {len(second_better)}, Tie: {len(equal_performance)}")


uniform_better, equal_performance, backward_better = compare_dfs(basebin, backwardbin, "result")
# uniform_better, equal_performance, backward_better = compare_dfs(uniformbin, backwardbin, "result")
print("=== Binary Classification Results ===")
print(f"Base Mean Metric: {basebin['result'].mean()}, Search Mean Metric {backwardbin['result'].mean()}")
print(f"Base Win: {len(uniform_better)}, Search Win: {len(backward_better)}, Tie: {len(equal_performance)}")
print(f"Base Win Tasks: {uniform_better}")
print(f"Search Win Tasks: {backward_better}\n")

uniform_better, equal_performance, backward_better = compare_dfs(basecat, backwardcat, "result")
# uniform_better, equal_performance, backward_better = compare_dfs(uniformcat, backwardcat, "result")
print("=== Multiclass Classification Results ===")
print(f"Base Mean Metric: {basecat['result'].mean()}, Search Mean Metric {backwardcat['result'].mean()}")
print(f"Base Win: {len(uniform_better)}, Search Win: {len(backward_better)}, Tie: {len(equal_performance)}")
print(f"Base Win Tasks: {uniform_better}")
print(f"Search Win Tasks: {backward_better}\n")

uniform_better, equal_performance, backward_better = compare_dfs(basereg, backwardreg, "result")
# uniform_better, equal_performance, backward_better = compare_dfs(uniformreg, backwardreg, "result")
print("=== Regression Results ===")
print(f"Base Mean Metric: {basereg['result'].mean()}, Search Mean Metric {backwardreg['result'].mean()}")
print(f"Base Win: {len(uniform_better)}, Search Win: {len(backward_better)}, Tie: {len(equal_performance)}")
print(f"Base Win Tasks: {uniform_better}")
print(f"Search Win Tasks: {backward_better}\n")
"""