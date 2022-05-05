# Adding a custom metric to AutoGluon
:label:`sec_tabularcustommetric`

**Tip**: If you are new to AutoGluon, review :ref:`sec_tabularquick` to learn the basics of the AutoGluon API.

This tutorial describes how to add a custom evaluation metric to AutoGluon that is used to inform validation scores, model ensembling, hyperparameter tuning, and more.

In this example, we show a variety of evaluation metrics and how to convert them to an AutoGluon Scorer, which can then be passed to AutoGluon models and predictors.

First, we will randomly generate 10 ground truth labels and predictions, and show how to calculate metric scores from them.


```{.python .input}
import numpy as np
y_true = np.random.randint(low=0, high=2, size=10)
y_pred = np.random.randint(low=0, high=2, size=10)

print(f'y_true: {y_true}')
print(f'y_pred: {y_pred}')
```

## Ensuring Metric is Serializable
You must define your custom metric in a separate python file that is imported for it to be serializable (able to be pickled).
If this is not done, AutoGluon will crash during fit when trying to parallelize model training with Ray.
In the below example, you would want to create a new python file such as `my_metrics.py` with `ag_accuracy_scorer` defined in it,
and then use it via `from my_metrics import ag_accuracy_scorer`.

If your metric is not serializable, you will get many errors similar to: `_pickle.PicklingError: Can't pickle`. Refer to https://github.com/awslabs/autogluon/issues/1637 for an example.

The custom metrics in this tutorial are **not** serializable for ease of demonstration. If `best_quality` preset was used, it would crash.

## Custom Accuracy Metric
We will start with calculating accuracy. A prediction is correct if the predicted value is the same as the true value, otherwise it is wrong.


```{.python .input}
import sklearn.metrics
sklearn.metrics.accuracy_score(y_true, y_pred)
```

Now, let's convert this evaluation metric to an AutoGluon Scorer.

We do this by calling `autogluon.core.metrics.make_scorer`.


```{.python .input}
from autogluon.core.metrics import make_scorer
ag_accuracy_scorer = make_scorer(name='accuracy',
                                 score_func=sklearn.metrics.accuracy_score,
                                 optimum=1,
                                 greater_is_better=True)
```

When creating the Scorer, we need to specify a name for the Scorer. This does not need to be any particular value, but is used when printing information about the Scorer during training.

Next, we specify the `score_func`. This is the function we want to wrap, in this case, sklearn's `accuracy_score` function.

We then need to specify the optimum value. This is necessary when calculating error as opposed to score. Error is calculated as `optimum - score`. It is also useful to identify when a score is optimal and cannot be improved.

Finally, we need to specify `greater_is_better`. In this case, `greater_is_better=True` because the best value returned is 1, and the worst value returned is less than 1 (0). It is very important to set this value correctly, otherwise AutoGluon will try to optimize for the **worst** model instead of the best.

Once created, the AutoGluon Scorer can be called in the same fashion as the original metric.


```{.python .input}
ag_accuracy_scorer(y_true, y_pred)
```

## Custom Mean Squared Error Metric

Next, let's show examples of how to convert regression metrics into Scorers.

First we generate random ground truth labels and their predictions, however this time they are floats instead of integers.


```{.python .input}
y_true = np.random.rand(10)
y_pred = np.random.rand(10)

print(f'y_true: {y_true}')
print(f'y_pred: {y_pred}')
```

A common regression metric is Mean Squared Error:


```{.python .input}
sklearn.metrics.mean_squared_error(y_true, y_pred)
```


```{.python .input}
ag_mean_squared_error_scorer = make_scorer(name='mean_squared_error',
                                           score_func=sklearn.metrics.mean_squared_error,
                                           optimum=0,
                                           greater_is_better=False)
```

In this case, optimum is 0 because this is an error metric.

Additionally, `greater_is_better=False` because sklearn reports error as positive values, and the lower the value is, the better.

A very important point about AutoGluon Scorers is that internally, they will always report scores in `greater_is_better=True` form. This means if the original metric was `greater_is_better=False`, AutoGluon's Scorer will flip the value. Therefore, error will be represented as negative values.

This is done to ensure consistency between different metrics.


```{.python .input}
ag_mean_squared_error_scorer(y_true, y_pred)
```

We can also specify metrics outside of sklearn. For example, below is a minimal implementation of mean squared error:


```{.python .input}
def mse_func(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return ((y_true - y_pred) ** 2).mean()

mse_func(y_true, y_pred)
```

All that is required is that the function take two arguments: `y_true`, and `y_pred` (or `y_pred_proba`), as numpy arrays, and return a float value.

With the same code as before, we can create an AutoGluon Scorer.


```{.python .input}
ag_mean_squared_error_custom_scorer = make_scorer(name='mean_squared_error',
                                                  score_func=mse_func,
                                                  optimum=0,
                                                  greater_is_better=False)
ag_mean_squared_error_custom_scorer(y_true, y_pred)
```

## Custom ROC AUC Metric

Here we show an example of a thresholding metric, `roc_auc`. A thresholding metric cares about the relative ordering of predictions, but not their absolute values.


```{.python .input}
y_true = np.random.randint(low=0, high=2, size=10)
y_pred_proba = np.random.rand(10)

print(f'y_true:       {y_true}')
print(f'y_pred_proba: {y_pred_proba}')
```


```{.python .input}
sklearn.metrics.roc_auc_score(y_true, y_pred_proba)
```

We will need to specify `needs_threshold=True` in order for downstream models to properly use the metric.


```{.python .input}
# Score functions that need decision values
ag_roc_auc_scorer = make_scorer(name='roc_auc',
                                score_func=sklearn.metrics.roc_auc_score,
                                optimum=1,
                                greater_is_better=True,
                                needs_threshold=True)
ag_roc_auc_scorer(y_true, y_pred_proba)
```

## Using Custom Metrics in TabularPredictor

Now that we have created several custom Scorers, let's use them for training and evaluating models.

For this tutorial, we will be using the Adult Income dataset.


```{.python .input}
from autogluon.tabular import TabularDataset

train_data = TabularDataset('https://autogluon.s3.amazonaws.com/datasets/Inc/train.csv')  # can be local CSV file as well, returns Pandas DataFrame
test_data = TabularDataset('https://autogluon.s3.amazonaws.com/datasets/Inc/test.csv')  # another Pandas DataFrame
label = 'class'  # specifies which column do we want to predict
train_data = train_data.sample(n=1000, random_state=0)  # subsample for faster demo

train_data.head(5)
```


```{.python .input}
from autogluon.tabular import TabularPredictor

predictor = TabularPredictor(label=label).fit(train_data, hyperparameters='toy')

predictor.leaderboard(test_data, silent=True)
```

We can pass our custom metrics into `predictor.leaderboard` via the `extra_metrics` argument:


```{.python .input}
predictor.leaderboard(test_data, extra_metrics=[ag_roc_auc_scorer, ag_accuracy_scorer], silent=True)
```

We can also pass our custom metric into the Predictor itself by specifying it during initialization via the `eval_metric` parameter:


```{.python .input}
predictor_custom = TabularPredictor(label=label, eval_metric=ag_roc_auc_scorer).fit(train_data, hyperparameters='toy')

predictor_custom.leaderboard(test_data, silent=True)
```

That's all it takes to create and use custom metrics in AutoGluon!

If you create a custom metric, consider [submitting a PR](https://github.com/awslabs/autogluon/pulls) so that we can add it officially to AutoGluon!

For a tutorial on implementing custom models in AutoGluon, refer to :ref:`sec_tabularcustommodel`.

For more tutorials, refer to :ref:`sec_tabularquick` and :ref:`sec_tabularadvanced`.
