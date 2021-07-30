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
        df1_rows = df1[df1["task"] == task]
        df2_rows = df2[df2["task"] == task]
        if len(df1_rows) > 0:
            df1_score = df1_rows[metric].dropna().mean()
        else:
            continue
        if len(df2_rows) > 0:
            df2_score = df2_rows[metric].dropna().mean()
        else:
            continue
        if df1_score > df2_score:
            df1_better.append(task)
        elif df1_score < df2_score:
            df2_better.append(task)
        else:
            equal_performance.append(task)
    return df1_better, equal_performance, df2_better


def limit_duration(df, duration=40000):
    return df[df['duration'] < duration]

"""
TODO
1. Debug by running single OpenML task with 1hr timeout on australian with a particular seed
"""

# base = pd.read_csv("~/Downloads/results_automlbenchmark_12h8c_autogluon.ag.12h8c.aws.20210728T082752.csv")
# uniform = pd.read_csv("~/Downloads/results_automlbenchmark_12h8c_autogluon_prune_uniform.ag.12h8c.aws.20210728T082754.csv")
base = pd.read_csv("~/Downloads/results_automlbenchmark_12h8c_autogluon_bestquality_norepeat.ag.12h8c.aws.20210728T084754.csv")
uniform = pd.read_csv("~/Downloads/results_automlbenchmark_12h8c_autogluon_prune_uniform_bestquality.ag.12h8c.aws.20210728T084850.csv")

task_metadata = pd.read_csv('result/task_metadata.csv')
base = add_dataset_info(base, task_metadata)
uniform = add_dataset_info(uniform, task_metadata)
# backward = add_dataset_info(backward, task_metadata)

DURATION = 43000
basebin = base[base["type"] == "binary"]
uniformbin = uniform[uniform["type"] == "binary"]
basecat = base[base["type"] == "multiclass"]
uniformcat = uniform[uniform["type"] == "multiclass"]
basereg = base[base["type"] == "regression"]
uniformreg = uniform[uniform["type"] == "regression"]
basedone = limit_duration(base, duration=DURATION)
uniformdone = limit_duration(uniform, duration=DURATION)

try:
    first_better, equal_performance, second_better = compare_dfs(base, uniform, "result")
    print(f"All Run Base Win: {len(first_better)}, Search Win: {len(second_better)}, Tie: {len(equal_performance)}")
    first_better, equal_performance, second_better = compare_dfs(basedone, uniformdone, "result")
    print(f"Finished Run Base Win: {len(first_better)}, Search Win: {len(second_better)}, Tie: {len(equal_performance)}")
    basebin_mean, basecat_mean, basereg_mean = round(basebin['result'].mean(), 4), round(basecat['result'].mean(), 4), round(basereg['result'].mean(), 4)
    print(f"Finished Base Results: (binary: {basebin_mean}), (multiclass: {basecat_mean}), (regression: {basereg_mean})")
    uniformbin_mean, uniformcat_mean, uniformreg_mean = round(uniformbin['result'].mean(), 4), round(uniformcat['result'].mean(), 4), round(uniformreg['result'].mean(), 4)
    print(f"Finished Search Results: (binary: {uniformbin_mean}), (multiclass: {uniformcat_mean}), (regression: {uniformreg_mean})")
except:
    import pdb; pdb.post_mortem()

import pdb; pdb.set_trace()
first_better, equal_performance, second_better = compare_dfs(base, uniform, "result")
print(f"Base Win: {len(first_better)}, Search Win: {len(second_better)}, Tie: {len(equal_performance)}")

"""
first_better, equal_performance, second_better = compare_dfs(base, uniform, "result")
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