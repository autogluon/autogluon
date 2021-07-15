# Fair Bayesian Optimization

Given the increasing importance of machine learning (ML) in our lives, with a wide use of automated systems in several domains, such as financial lending, hiring, criminal justice, and college admissions, there has been a major concern for ML to unintentionally encode societal biases and result in systematic discrimination. For example, a classifier that is only tuned to maximize prediction accuracy may unfairly predict a high credit risk for some subgroups of the population applying for a loan. 

In several real-world domains, accuracy is not the only objective of interest. The model should simultaneously give guarantees on other important aspects. Algorithmic fairness tries to find algorithms that not only keep a high level of accuracy but also enforce a certain level of fairness and avoid biases.

In this tutorial we are going to use the German Credit Data from the [UCI Machine Learning Repository](http://archive.ics.uci.edu/ml). This dataset contains a binary classification task, where the goal is to predict if a person has "good" or "bad" credit risk.


```python
# AutoGluon and HPO tools
import autogluon.core as ag
import pandas as pd
import numpy as np
import random

# Plots
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

# Fixing seed for reproducibility
SEED = 999
random.seed(SEED)
np.random.seed(SEED)
```

## Dataset setup and visualization

This code will automatically download the data from the UCI repository.
You can download the data manually from here: [UCI german lending dataset link.](https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data))


```python
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data', header=None, sep=' ')
df.columns = ["CheckingAC_Status","MaturityMonths","CreditHistory","Purpose","LoanAmount","SavingsAC",
                  "Employment","InstalmentPctOfIncome","SexAndStatus","OtherDebts","PresentResidenceYears",
                  "Property","Age","OtherInstalmentPlans","Housing","NumExistingLoans","Job",
                  "Dependents","Telephone","ForeignWorker","Class1Good2Bad"]
```

We are now ready to preprocess the raw features as following:


```python
df["target"] = df["Class1Good2Bad"].replace([1, 2], [1, 0]).astype("category")
df = df.drop(columns=["Class1Good2Bad"])
df["CheckingAC_Status"] = (
    df["CheckingAC_Status"]
    .replace(["A11", "A12", "A13", "A14"], ["x < 0 DM", "0 <= x < 200 DM", "x >= 200DM", "no checking account"])
    .astype("category")
)
df["CreditHistory"] = (
    df["CreditHistory"]
    .replace(
        ["A30", "A31", "A32", "A33", "A34"],
        ["no credits", "all credits paid", "existing credits paid", "delay", "critical accnt. / other credits"],
    )
    .astype("category")
)
df["Purpose"] = (
    df["Purpose"]
    .replace(
        ["A40", "A41", "A42", "A43", "A44", "A45", "A46", "A47", "A48", "A49", "A410"],
        [
            "new car",
            "used car",
            "forniture",
            "radio/tv",
            "appliances",
            "repairs",
            "education",
            "vacation",
            "retraining",
            "business",
            "others",
        ],
    )
    .astype("category")
)
df["SavingsAC"] = (
    df["SavingsAC"]
    .replace(
        ["A61", "A62", "A63", "A64", "A65"],
        ["x < 100 DM", "100 <= x < 500 DM", "500 <= x < 1000 DM", "x >= 1000 DM", "unknown"],
    )
    .astype("category")
)
df["Employment"] = (
    df["Employment"]
    .replace(
        ["A71", "A72", "A73", "A74", "A75"],
        ["unemployed", "x < 1 year", "1 <= x < 4 years", "4 <= x < 7 years", "x >= 7 years"],
    )
    .astype("category")
)
df["SexAndStatus"] = (
    df["SexAndStatus"]
    .replace(
        ["A91", "A92", "A93", "A94", "A95"],
        [
            "male divorced/separated",
            "female divorced/separated/married",
            "male single",
            "male married/widowed",
            "female single",
        ],
    )
    .astype("category")
)
df["OtherDebts"] = (
    df["OtherDebts"].replace(["A101", "A102", "A103"], ["none", "co-applicant", "guarantor"]).astype("category")
)
df["Property"] = (
    df["Property"]
    .replace(
        ["A121", "A122", "A123", "A124"],
        ["real estate", "soc. savings / life insurance", "car or other", "unknown"],
    )
    .astype("category")
)
df["OtherInstalmentPlans"] = (
    df["OtherInstalmentPlans"].replace(["A141", "A142", "A143"], ["bank", "stores", "none"]).astype("category")
)
df["Housing"] = df["Housing"].replace(["A151", "A152", "A153"], ["rent", "own", "for free"]).astype("category")
df["Job"] = (
    df["Job"]
    .replace(
        ["A171", "A172", "A173", "A174"],
        [
            "unemployed / unskilled-non-resident",
            "unskilled-resident",
            "skilled employee / official",
            "management / self-employed / highly qualified employee / officer",
        ],
    )
    .astype("category")
)
df["Telephone"] = df["Telephone"].replace(["A191", "A192"], ["none", "yes"]).astype("category")
df["ForeignWorker"] = df["ForeignWorker"].replace(["A201", "A202"], ["yes", "no"]).astype("category")
```

We can plot a few statistics to see if there is a risk of bias, due to unbalanced data or other factors. Let's begin with checking the data we loaded.


```python
df
```

We can now plot the histograms for:

(1) binary target output for good (1) or bad (0) credit risk;

(2) binary ForeignWorker feature;

(3) binary target output for individuals with ForeignWorker feature equals to "yes";

(4) binary target output for individuals with ForeignWorker feature equals to "no".


```python
fig, (ax1, ax2) = plt.subplots(ncols=2)
fig.tight_layout(pad=5.0)

sns.countplot(df['target'], ax=ax1)
ax1.set_title('Target')

sns.countplot(df['ForeignWorker'], ax=ax2)
ax2.set_title('ForeignWorker')
plt.show()
```

We can note that our dataset unbalanced, having more than double positive examples compared to negative ones. The unbalance is ever higher concerning the two subgroups of foreign against local workers.


```python
fig, (ax1, ax2) = plt.subplots(ncols=2)
fig.tight_layout(pad=5.0)

sns.countplot(df[df["ForeignWorker"] == 'no']['target'], ax=ax1)
ax1.set_title('Target | Non ForeignWorker')


sns.countplot(df[df["ForeignWorker"] == 'yes']['target'], ax=ax2)
ax2.set_title('Target | ForeignWorker')
plt.show()
```

Finally, here we have a possible source of bias: the proportion between positive and negative target label is different between the two subgroups.


```python
bad_credit_risk_foreign = 1.0 - np.array(df[df["ForeignWorker"] == 'yes']['target']).mean()
bad_credit_risk_local = 1.0 - np.array(df[df["ForeignWorker"] == 'no']['target']).mean()

print(f"Proportion of bad credit risk for foreign workers: {bad_credit_risk_foreign}")
print(f"Proportion of bad credit risk for non foreign workers: {bad_credit_risk_local}")
```

## Fairness metrics

We are now interested in measuring the unfairness (bias) of our model $f$ with respect to the sensitive attribute gender (i.e., _is our model discriminating a specific group of people?_). It is important to note that  there is no consensus on a unique definition of fairness of a ML model. In fact, some of the most common definitions are even conflicting.

In this tutorial we decided to select a commonly used measure for bias called Statistical Parity (SP). SP for a binary classification problem requires the trained model to have the same probability of predicting a positive label among the different subgourps $A$ and $B$. Empirically, we are interested in controlling the amount of violation of this constraint, and we define the Difference in Statistical Parity (DSP). Ideally, we would like this quantity to be small than a certain threshold value $\epsilon$:

$$DSP(f) = \Big| \mathbb{P}_{(x,y)} \big[ f(x)>0 \, \big| \, x \text{ in group } A \big] -  \mathbb{P}_{(x,y)} \big[ f(x)>0 \, \big| \, x \text{ in group } B \big] \Big| \leq \epsilon.$$

Another possible definition is called [Equal Opportunity](https://ai.googleblog.com/2016/10/equality-of-opportunity-in-machine.html) (EO), defined as the property of having a similar True Positive Rate among the different groups. Also in this case we can define the Difference in Equal Opportunity (DEO), and our goal is to keep it smaller that $\epsilon$:

$$DEO(f) = \Big| \mathbb{P}_{(x,y)} \big[ f(x)>0 \, \big| \, x \text{ in group } A, y = 1 \big] -  \mathbb{P}_{(x,y)} \big[ f(x)>0 \, \big| \, x \text{ in group } B, y = 1 \big] \Big| \leq \epsilon$$


A lower value of these measures means a fairer model $f$ with respect to the sensitive feature.


```python
from sklearn.metrics import accuracy_score

def DSP(model, X, Y, groups):
    # model: the trained model
    # X: our data of n examples with d features
    # Y: binary labels of our n examples (1 = positive)
    # groups: a list of n values binary values defining two different subgroups of the populations
    
    fY = model.predict(X) 
    sp = [0, 0]
    sp[0] = float(len([1 for idx, fy in enumerate(fY) if fy == 1 and groups[idx] == 0])) / len([1 for idx, fy in enumerate(fY) if groups[idx] == 0])
    sp[1] = float(len([1 for idx, fy in enumerate(fY) if fy == 1 and groups[idx] == 1])) / len([1 for idx, fy in enumerate(fY) if groups[idx] == 1])
    return abs(sp[0] - sp[1])

def DEO(model, X, Y, groups):
    # model: the trained model
    # X: our data of n examples with d features
    # Y: binary labels of our n examples (1 = positive)
    # groups: a list of n values binary values defining two different subgroups of the populations
    
    fY = model.predict(X) 
    eo = [0, 0]
    eo[0] = float(len([1 for idx, fy in enumerate(fY) if fy == 1 and groups[idx] == 0 and Y[idx] == 1])) / len([1 for idx, fy in enumerate(fY) if groups[idx] == 0 and Y[idx] == 1])
    eo[1] = float(len([1 for idx, fy in enumerate(fY) if fy == 1 and groups[idx] == 1 and Y[idx] == 1])) / len([1 for idx, fy in enumerate(fY) if groups[idx] == 1 and Y[idx] == 1])
    return abs(eo[0] - eo[1])
```

Encoding of the categorical features of our dataset:


```python
to_hot_encode = ['CheckingAC_Status', 'CreditHistory', 'Purpose', 'SavingsAC', 'Employment', 'SexAndStatus', 'OtherDebts', 'Property',  'OtherInstalmentPlans', 'Housing', 'Job', 'Telephone', 'ForeignWorker']
coded_df = pd.concat((df[to_hot_encode],
                      pd.get_dummies(df, columns=to_hot_encode, drop_first=True)),
                     axis=1)

for col in to_hot_encode:
    coded_df.pop(col)

coded_df
```

## Standard Bayesian Optimization

We choose a RandomForest classifier as our base ML model for this task, and we run standard BO to optimize the hyperparameters of it. While we optimize the accuracy of our model, we keep track of its DSP.


```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

Y = coded_df.pop('target').values
X = coded_df.values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
print('X shape:', X.shape)
print('Y shape:', Y.shape)
```


```python
def process_training_history(task_dicts, start_timestamp, 
                             runtime_fn=None):
    task_dfs = []
    for task_id in task_dicts:
        task_df = pd.DataFrame(task_dicts[task_id])
        error = 1.0 - task_df["objective"]
        is_fair = (task_df["constraint_metric"] < 0.0).values
        if is_fair:
            fair_error = error
        else:
            fair_error = 1.0 # worst possible value
        task_df = task_df.assign(task_id=task_id,
                                 error=error,
                                 fair_error=fair_error,
                                 is_fair=is_fair)
        task_dfs.append(task_df)

    result = pd.concat(task_dfs, axis="index", ignore_index=True, sort=True)
    # calculate incumbent best -- the cumulative minimum of the error.
    result = result.assign(best=result["error"].cummin())
    result = result.assign(fair_best=result["fair_error"].cummin())
    return result
```


```python
ACTIVE_METRIC_NAME = 'obj_metric'
CONSTRAINT_METRIC_NAME = 'constr_metric'
REWARD_ATTR_NAME = 'objective'
FAIRNESS_THRESHOLD = 0.01
FAIRNESS_DEFINITION = DSP  # You can use any fairness definition, such as DSP or DEO
```

We tune RF on a 3-dimensional search space: 

* min_samples_split in [0.01, 0.5] (log scaled)
* tree maximum depth in {1, 2, 3, 4, 5}
* criterion for quality of split in {Gini, Entropy}


```python
def create_train_fn_constraint(fairness_threshold, fairness_definition):
    @ag.args(min_samples_split=ag.space.Real(lower=0.01, upper=1.0, log=True),
             max_depth=ag.space.Int(lower=1, upper=50),
             criterion=ag.space.Categorical('gini', 'entropy')
            )
    def run_opaque_box(args, reporter):
        opaque_box_eval = opaque_box(args.min_samples_split,
                                     args.max_depth, 
                                     args.criterion, 
                                     fairness_threshold,
                                     fairness_definition)
        reporter(objective=opaque_box_eval[ACTIVE_METRIC_NAME],
                 constraint_metric=opaque_box_eval[CONSTRAINT_METRIC_NAME])
    return run_opaque_box

run_opaque_box = create_train_fn_constraint(fairness_threshold=FAIRNESS_THRESHOLD, 
                                            fairness_definition=FAIRNESS_DEFINITION)  
# fairness constraint: DSP < 0.01
```


```python
def opaque_box(min_samples_split, max_depth, criterion, fairness_threshold, fairness_definition):
    classifier = RandomForestClassifier(max_depth=max_depth,
                                        min_samples_split=min_samples_split,
                                        criterion=criterion)
    classifier.fit(X_train, Y_train)
    DSP_ForeignWorker = fairness_definition(classifier, X_test, Y_test, df["ForeignWorker"].values == "yes")
    accuracy = accuracy_score(classifier.predict(X_test), Y_test)
    evaluation_dict = {}
    evaluation_dict[ACTIVE_METRIC_NAME] = accuracy
    evaluation_dict[CONSTRAINT_METRIC_NAME] = DSP_ForeignWorker - fairness_threshold  # If DSP < fairness threshold, then fair
    return evaluation_dict
```


```python
# Create scheduler and searcher:

# First get_config are random, the remaining ones use constrained BO
search_options = {
    'random_seed': SEED,
    'num_fantasy_samples': 5,
    'num_init_random': 1,
    'debug_log': True}
myscheduler = ag.scheduler.FIFOScheduler(
    run_opaque_box,
    searcher='bayesopt',
    search_options=search_options,
    num_trials=15,
    reward_attr=REWARD_ATTR_NAME
)

# Run HPO experiment
myscheduler.run()
myscheduler.join_jobs()
```


```python
results_df_standard = process_training_history(myscheduler.training_history.copy(),
                                               start_timestamp=myscheduler._start_time)
results_df_standard.head()
```

Let's look at the empirical probability that standard BO finds a fair model (with respect to the constraint of DPS < 0.01), the average DSP and classification error.


```python
print('Average DSP (unfairness):', np.mean(results_df_standard['constraint_metric'] + FAIRNESS_THRESHOLD))
print('Average classification error:', np.mean(results_df_standard['best']))
```

A higher value of DSP can potentially highlight a discriminatory behavior of our model. 

A possible way to be more effective in finding unbiased models is to search for set of hyperparamters that is able to generate a model that is both fair and accurate. A solution to find accurate model under fairness constraints is provided by the Constrained Bayesian Optimization framework (CBO). This technique allows us to specify any constraint, such as “DSP < 0.1” and searching for hyperparameters able to generate accurate models such that the constrained is not violated. 

## Constrained Bayesian Optimization

In our example, using German Credit data we can now easily run CBO trying to find the most accurate model such that DSP<0.1. Also in this case we selected a Random Forest as our base ML model.


```python
run_opaque_box = create_train_fn_constraint(fairness_threshold=FAIRNESS_THRESHOLD, 
                                            fairness_definition=FAIRNESS_DEFINITION)  
# fairness constraint: DSP < 0.01
```


```python
# Create scheduler and searcher:

# First get_config are random, the remaining ones use constrained BO
search_options = {
    'random_seed': SEED,
    'num_fantasy_samples': 5,
    'num_init_random': 1,
    'debug_log': True}
myscheduler = ag.scheduler.FIFOScheduler(
    run_opaque_box,
    searcher='constrained_bayesopt',
    search_options=search_options,
    num_trials=15,
    reward_attr=REWARD_ATTR_NAME,
    constraint_attr='constraint_metric'
)

# Run HPO experiment
myscheduler.run()
myscheduler.join_jobs()
```


```python
results_df = process_training_history(myscheduler.training_history.copy(),
                                      start_timestamp=myscheduler._start_time)
results_df.head()
```

Let's see the empirical probability that the *constrained* BO procedure finds a fair model (with respect to the constraint of DPS < 0.01), the average DSP and classification error.


```python
print('Average DSP (unfairness):', np.mean(results_df['constraint_metric'] + FAIRNESS_THRESHOLD))
print('Average classification error:', np.mean(results_df['error']))
```

FairBO is able to better focus on the fair region of the search space.

## Standard vs CBO

The presented method is model-agnostic and is able to handle any statistical notion of fairness.


```python
for method_name in ['BO', 'CBO']:
    if method_name == 'BO':
        df = results_df_standard
    else:
        df = results_df
    unfairness = list(df['constraint_metric'] + FAIRNESS_THRESHOLD)
    accuracies = list(1.0 - df['error'])
    ub = FAIRNESS_THRESHOLD
    scaling = 1
    best_acc = 0.0
    unf_best = 1.0
    for acc, unf in zip(accuracies, unfairness):
        if unf <= ub * scaling and acc > best_acc:
            best_acc = acc
            unf_best = unf
    cmap = sns.cubehelix_palette(as_cmap=True)
    f, ax = plt.subplots()
    points = ax.scatter(x=accuracies, y=unfairness, c=list(range(len(accuracies))), alpha=0.5, cmap=cmap)
    plt.scatter(best_acc, unf_best, color="red", marker='*', label='Fair best', s=100, alpha=1)
    plt.xlim(0.67, 0.78)
    plt.ylim(-0.01, 0.115)

    plt.legend(loc='upper right')
    plt.axhline(y=ub * scaling, xmin=0, xmax=1, color='black', linestyle='--', alpha=1, linewidth=1)

    f.colorbar(points, label='iteration')

    
    plt.xlabel('validation accuracy')

    plt.ylabel(f'DSP')
    algorithm_name = 'Random Forest'
    dataset_name = 'German'
    plt.title(f'{method_name} ({algorithm_name} on {dataset_name})')
```

In the plots above, the horizontal line is the fairness constraint,
set to DSP ≤ 0.01, and darker dots correspond to later BO iterations. Standard BO can get stuck in high-performing yet unfair
regions, failing to return a well-performing, feasible solution. CBO is able to focus the exploration over the fair area of the hyperparameter space, and finds a more accurate fair solution.

The presented method is model-agnostic and is able to handle any statistical notion of fairness. For instance, you can
repeat the experiments plugging in a constraint on DEO instead of DSP.
