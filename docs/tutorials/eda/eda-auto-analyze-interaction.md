# Feature Interaction Charting

This tool is made for quick interactions visualization between variables in a dataset. User can specify the variables to
be plotted on the x, y and hue (color) parameters. The tool automatically picks chart type to render based on the
detected variable types and renders 1/2/3-way interactions.

This feature can be useful in exploring patterns, trends, and outliers and potentially identify good predictors for the
task.

## Using Interaction Charts for Missing Values Filling

Let's load titanic dataset:

```{.python .input}
import pandas as pd

df_train = pd.read_csv('https://autogluon.s3.amazonaws.com/datasets/titanic/train.csv')
df_test = pd.read_csv('https://autogluon.s3.amazonaws.com/datasets/titanic/test.csv')
target_col = 'Survived'
```

Next we will look at missing data in the variable:

```{.python .input}
import autogluon.eda.analysis as eda
import autogluon.eda.visualization as viz
import autogluon.eda.auto as auto

df_all = pd.concat([df_train, df_test], ignore_index=True).drop(columns=target_col)
auto.analyze(train_data=df_all, anlz_facets=[
    eda.missing.MissingValuesAnalysis(),
], viz_facets=[
    viz.DatasetStatistics(),
    viz.missing.MissingValues(graph_type='matrix', figsize=(12, 6)),
])
```

It looks like there are only two null values in the `Embarked` feature. Let's see what are those two null values:

```{.python .input}
df_all[df_all.Embarked.isna()]
```

We may be able to fill these by looking at other independent variables. Both passengers paid a `Fare` of `$80`, are
of `Pclass` `1` and `female` `Sex`. Let's see how the `Fare` is distributed among all `Pclass` and `Embarked` feature
values:

```{.python .input}
auto.analyze_interaction(train_data=df_all, x='Embarked', y='Fare', hue='Pclass')
```

The average `Fare` closest to `$80` are in the `C` `Embarked` values where `Pclass` is `1`. Let's fill in the missing
values as `C`.

## Using Interaction Charts To Learn Information About the Data

```{.python .input}
auto.analyze_interaction(x='Pclass', y='Survived', train_data=df_train, test_data=df_test)
```

It looks like `63%` first class passengers survived; `48%` second class and only `24%` third class passenger survived.
Similar information is visible via `Fare` variable:

```{.python .input}
auto.analyze_interaction(x='Fare', hue='Survived', train_data=df_train, test_data=df_test, chart_args=dict(fill=True))
```

```{.python .input}
auto.analyze_interaction(x='Age', hue='Survived', train_data=df_train, test_data=df_test)
```

The very left part of the distribution on this chart possibility hints that children and infants were the priority.

```{.python .input}
auto.analyze_interaction(x='Fare', y='Age', hue='Survived', train_data=df_train, test_data=df_test)
```

This chart highlights three outliers with a Fare of over `\$500`. Let's take a look at these:
```{.python .input}
df_all[df_all.Fare > 400]
```
As you can see all 4 passengers share the same ticket. Per-person fare would be 1/4 of this value. Looks like we can 
add a new feature to the dataset fare per person; also this allows us to see if some passengers travelled in larger 
groups. Let's create two new features and take at the Fare-Age relationship once again.

```{.python .input}
ticket_to_count = df_all.groupby(by='Ticket')['Embarked'].count().to_dict()
data = df_train.copy()
data['GroupSize'] = data.Ticket.map(ticket_to_count)
data['FarePerPerson'] = data.Fare / data.GroupSize

auto.analyze_interaction(x='FarePerPerson', y='Age', hue='Survived', train_data=data)
auto.analyze_interaction(x='FarePerPerson', y='Age', hue='Pclass', train_data=data)
```

You can see cleaner separation between `Fare`, `Pclass` and `Survived` now.
