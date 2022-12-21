# Interpretable rule-based modeling
:label:`sec_tabularinterpretability`

*Note*: This addition was made through collaboration with [the Yu Group](https://www.stat.berkeley.edu/~yugroup/) of Statistics and EECS from UC Berkeley.

**Tip**: Prior to reading this tutorial, it is recommended to have a basic understanding of the TabularPredictor API covered in :ref:`sec_tabularquick`.

In this tutorial, we will explain how to automatically use interpretable models powered by integration with [🔍 the imodels package](https://github.com/csinva/imodels). This allows for automatically learning models based on rules which are extremely concise and can be useful for (1) understanding data or (2) building a transparent predictive model.

*Note*: `imodels` must be installed for this tutorial. You can ensure `imodels` is installed via `pip install autogluon.tabular[imodels]`. `imodels` is not installed by default.

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

The rule-based models take slightly different forms (see below), but all try to optimize predictive performance using as few rules as possible. See the [imodels package](https://github.com/csinva/imodels) for more details.

 <img align="center" width=60% src="https://csinva.io/imodels/img/imodels_logo.svg?sanitize=True"/>

![](https://raw.githubusercontent.com/csinva/imodels/master/docs/img/model_table_rules.png)

Specifically, the interpretable preset fits different hyperparameter configurations of 5 models types:
1. Greedy CART decision tree - returns a tree learned via greedy optimization ([Description](https://scikit-learn.org/stable/modules/tree.html#tree), [Paper](https://www.taylorfrancis.com/books/mono/10.1201/9781315139470/classification-regression-trees-leo-breiman-jerome-friedman-richard-olshen-charles-stone))
2. Hierarchical Shrinkage tree - returns regularized version of CART decision tree ([Description](https://csinva.io/imodels/shrinkage.html), [Paper](https://arxiv.org/abs/2202.00858))
3. Fast interpretable greedy-tree sum - returns a *sum* of trees, which are greedily grown simultaneously ([Description](https://csinva.io/imodels/figs.html), [Paper](https://arxiv.org/abs/2202.00858))
4. RuleFit - returns a set of weighted rules, which are learned by a sparse linear model on rules extracted from decision trees ([Description](https://christophm.github.io/interpretable-ml-book/rulefit.html), [Paper](https://arxiv.org/abs/0811.1679))
5. Boosted rule set - returns a set of rules, which are learned sequentially via AdaBoost ([Description](https://scikit-learn.org/stable/modules/ensemble.html#adaboost), [Paper](https://www.sciencedirect.com/science/article/pii/S002200009791504X)) 


In addition to the usual functions in `TabularPredictor`, this predictor fitted with interpretable models has some additional functionality. For example, we can now inspect the complexity of the fitted models (i.e. how many rules they contain).

```{.python .input}
predictor.interpretable_models_summary()
```

We can also explicitly inspect the rules of the best-performing model.

```{.python .input}
predictor.print_interpretable_rules() # can optionally specify a model name or complexity threshold
```

In some cases, these rules are sufficient to accurately make predictions. In other cases, they may just be used to gain a better understanding of the data before proceeding with more black-box models.
