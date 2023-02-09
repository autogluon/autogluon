# Automated Dataset Overview

In this section we explore automated dataset overview functionality. This feature allows you to easily get
a high-level understanding of datasets, including information about the number of rows and columns, the data types
of each column, and basic statistical information such as min/max values, mean, quartiles, and standard deviation. This
functionality can be a valuable tool for quickly identifying potential issues or areas of interest in your dataset
before diving deeper into your analysis.

Additionally, this feature also provides graphical representations of distances between features to highlight features
that can be either simplified or completely removed. For each detected near-duplicate group, it plots interaction charts
so it can be inspected visually.

## Example

We will start with getting titanic dataset and performing a quick one-line overview to get the information.

```{.python .input}
import pandas as pd

df_train = pd.read_csv('https://autogluon.s3.amazonaws.com/datasets/titanic/train.csv')
df_test = pd.read_csv('https://autogluon.s3.amazonaws.com/datasets/titanic/test.csv')
target_col = 'Survived'
```

To showcase near duplicates detection functionality, let's add a duplicated column: 
```{.python .input}
df_train['Fare_duplicate'] = df_train['Fare']
df_test['Fare_duplicate'] = df_test['Fare']
```

The report consists of multiple parts: statistical information overview enriched with feature types detection and
missing value counts.

The last chart is a feature distance. It measures the similarity between features in a dataset. For example, if two
variables are almost identical, their feature distance will be small. Understanding feature distance is useful in feature
selection, where it can be used to identify which variables are redundant and should be considered for removal. To
perform the analysis, we need just one line:

```{.python .input}
import autogluon.eda.auto as auto

auto.dataset_overview(train_data=df_train, test_data=df_test, label=target_col)
```

