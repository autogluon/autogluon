# Predicting Columns in a Table - Quick Start
:label:`sec_tabularquick`

Via a simple `fit()` call, AutoGluon can produce highly-accurate models to predict the values in one column of a data table based on the rest of the columns' values. Use AutoGluon with tabular data for both classification and regression problems. This tutorial demonstrates how to use AutoGluon to produce a classification model that predicts whether or not a person's income exceeds $50,000.

To start, import autogluon and TabularPrediction module as your task:

```{.python .input}
import autogluon as ag
from autogluon import TabularPrediction as task
```

Load training data from a CSV file into an AutoGluon Dataset object. This object is essentially equivalent to a [Pandas DataFrame](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html) and the same methods can be applied to both.

```{.python .input}
train_data = task.Dataset(file_path='https://autogluon.s3.amazonaws.com/datasets/Inc/train.csv')
train_data = train_data.head(500) # subsample 500 data points for faster demo
print(train_data.head())
```

Note that we loaded data from a CSV file stored in the cloud (AWS s3 bucket), but you can you specify a local file-path instead if you have already downloaded the CSV file to your own machine (e.g., using `wget`).
Each row in the table `train_data` corresponds to a single training example. In this particular dataset, each row corresponds to an individual person, and the columns contain various characteristics reported during a census.

Let's first use these features to predict whether the person's income exceeds $50,000 or not, which is recorded in the `class` column of this table.

```{.python .input}
label_column = 'class'
print("Summary of class variable: \n", train_data[label_column].describe())
```

Now use AutoGluon to train multiple models:

```{.python .input}
dir = 'agModels-predictClass' # specifies folder where to store trained models
predictor = task.fit(train_data=train_data, label=label_column, output_directory=dir)
```

Next, load separate test data to demonstrate how to make predictions on new examples at inference time:

```{.python .input}
test_data = task.Dataset(file_path='https://autogluon.s3.amazonaws.com/datasets/Inc/test.csv')
y_test = test_data[label_column]  # values to predict
test_data_nolab = test_data.drop(labels=[label_column],axis=1) # delete label column to prove we're not cheating
print(test_data_nolab.head())
```

We use our trained models to make predictions on the new data and then evaluate performance:

```{.python .input}
predictor = task.load(dir) # unnecessary, just demonstrates how to load previously-trained predictor from file

y_pred = predictor.predict(test_data_nolab)
print("Predictions:  ", y_pred)
perf = predictor.evaluate_predictions(y_true=y_test, y_pred=y_pred, auxiliary_metrics=True)
```

Now you're ready to try AutoGluon on your own tabular datasets!
As long as they're stored in a popular format like CSV, you should be able to achieve strong predictive performance with just 2 lines of code:

```
from autogluon import TabularPrediction as task
predictor = task.fit(train_data=task.Dataset(file_path=<file-name>), label_column=<variable-name>)
```


## Description of fit():

Here we discuss what happened during `fit()`.

Since there are only two possible values of the `class` variable, this was a binary classification problem, for which an appropriate performance metric is *accuracy*. AutoGluon automatically infers this as well as the type of each feature (i.e., which columns contain continuous numbers vs. discrete categories). AutogGluon can also automatically handle common issues like missing data and rescaling feature values.

We did not specify separate validation data and so AutoGluon automatically choses a random training/validation split of the data. The data used for validation is seperated from the training data and is used to determine the models and hyperparameter-values that produce the best results.  Rather than just a single model, AutoGluon trains multiple models and ensembles them together to ensure superior predictive performance.

By default, AutoGluon tries to fit various types of models including neural networks and tree ensembles. Each type of model has various hyperparameters, which traditionally, the user would have to specify.
AutoGluon automates this process.

AutoGluon automatically and iteratively tests values for hyperparameters to produce the best performance on the validation data. This involves repeatedly training models under different hyperparameter settings and evaluating their performance. This process can be computationally-intensive, so `fit()` can parallelize this process across multiple threads (and machines if distributed resources are available). To control runtimes, you can specify various arguments in fit() as demonstrated in the subsequent **In-Depth** tutorial.

For tabular problems, `fit()` returns a `Predictor` object. Besides inference, this object can also be used to view a summary of what happened during fit.

```{.python .input}
results = predictor.fit_summary()
```

From this summary, we can see that AutoGluon trained many different types of models as well as an ensemble of the best-performing models.  The summary also describes the actual models that were trained during fit and how well each model performed on the held-out validation data.  We can also view what properties AutoGluon automatically inferred about our prediction task:

```{.python .input}
print("AutoGluon infers problem type is: ", predictor.problem_type)
print("AutoGluon categorized the features as: ", predictor.feature_types)
```

AutoGluon correctly recognized our prediction problem to be a binary classification task and decided that variables such as `age` should be represented as integers, whereas variables such as `workclass` should be represented as categorical objects.

## Regression (predicting numeric table columns):

To demonstrate that `fit()` can also automatically handle regression tasks, we now try to predict the numeric `age` variable in the same table based on the other features:

```{.python .input}
age_column = 'age'
print("Summary of age variable: \n", train_data[age_column].describe())
```

We again call `fit()`, imposing a time-limit this time (in seconds), and also demonstrate a shorthand method to evaluate the resulting model on the test data (which contain labels):

```{.python .input}
predictor_age = task.fit(train_data=train_data, output_directory="agModels-predictAge", label=age_column, time_limits=60)
performance = predictor_age.evaluate(test_data)
```

Note that we didn't need to tell AutoGluon this is a regression problem, it automatically inferred this from the data and reported the appropriate performance metric (RMSE by default). To specify a particular evaluation metric other than the default, set the `eval_metric` argument of `fit()` and AutoGluon will tailor its models to optimize your metric (e.g. `eval_metric = 'mean_absolute_error'`). For evaluation metrics where higher values are worse (like RMSE), AutoGluon may sometimes flips their sign and print them as negative values during training (as it internally assumes higher values are better).
