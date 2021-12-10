# Interpretable rule-based modeling
:label:`sec_tabularinterpretability`

*Note*: This addition was made through collaboration with [the Yu-Group](https://www.stat.berkeley.edu/~yugroup/) at UC Berkeley.

**Tip**: Prior to reading this tutorial, it is recommended to have a basic understanding of the TabularPredictor API covered in :ref:`sec_tabularquick`.

In this tutorial, we will explain how to automatically use interpretable models powered by integration with [🔍 the imodels package](https://github.com/csinva/imodels). This allows for automatically learning models based on rules which are extremely concise and can be useful for (1) understanding data or (2) building a transparent predictive model.

Begin by loading in data to predict. Note: interpretable rule-based modeling is currently only supported for binary classification.

```{.python .input}
from autogluon.tabular import TabularDataset, TabularPredictor
train_data = TabularDataset('https://autogluon.s3.amazonaws.com/datasets/Inc/train.csv')
subsample_size = 500  # subsample subset of data for faster demo, try setting this to much larger values
train_data = train_data.sample(n=subsample_size, random_state=0)
train_data.head()
```

Now, we create a predictor and fit it to the data. By specifying `presets='interpretable'`, we tell the predictor to fit only interpretable models.

```{.python .input}
predictor = TabularPredictor(label='class')
predictor.fit(train_data, presets='interpretable')
predictor.leaderboard()
```

The rule-based models take slightly different forms (see below), but all try to optimize predictive performance using as few rules as possible. See [imodels package](https://github.com/csinva/imodels) for more details.

 <img align="center" width=60% src="https://csinva.io/imodels/img/imodels_logo.svg?sanitize=True"/>

![](https://raw.githubusercontent.com/csinva/imodels/master/docs/img/model_table_rules.png)

In addition to the usual functions in `TabularPredictor`, this predictor fitted with interpretable models has some additional functionality. For example, we can now inspect the complexity of the fitted models (i.e. how many rules they contain).

```{.python .input}
predictor.interpretable_models_summary()
```

We can also explicitly inspect the rules of the best-performing model.

```{.python .input}
predictor.print_interpretable_rules() # can optionally specify a model name or complexity threshold
```

In some cases, these rules are sufficient to accurately make predictions. In other cases, they may just be used to gain a better understanding of the data before proceeding with more black-box models.