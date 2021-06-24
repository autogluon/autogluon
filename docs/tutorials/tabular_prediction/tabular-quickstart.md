# Predicting a Column in a Table
:label:`sec_tabularquick`

This tutorial shows the basic usage of AutoGluon to predict a column in a table. To start, let's first import both `TabularPredictor` and `TabularDataset` classes from the `tabular` module. 

```{.python .input}
from autogluon.tabular import TabularDataset, TabularPredictor
```

## Loading Data

Next let's load the dataset from a [CSV file](https://en.wikipedia.org/wiki/Comma-separated_values) through `TabularDataset`. A CSV file is a delimited text file that uses a comma to separate values. We assume each row contains an example. In a row, features and labels are separated by commas. We pass the URL of the CSV file into `TabularDataset`, it will download and load the file. If your data is ready on local disk, you can just pass the file path.


```{.python .input}
train_data = TabularDataset('https://autogluon.s3.amazonaws.com/datasets/Inc/train.csv')
# subsample subset of data for faster demo
train_data = train_data.sample(n=500)
train_data.head(n=2)
```

If you are familiar with [Pandas DataFrame](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html), you will find `TabularDataset` is a subclass of `DataFrame`, so all pandas functions can be applied to `TabularDataset` directly. 

Using the `head` method we print the first two examples. You can see each feature is either a number or a category presented by a string. The last column, named `class`, contains the label. By describing the label column, we can see it has 2 unique values. So our goal is to predict if the person's income exceeds $50,000 or not.

```{.python .input}
label = 'class'
train_data[label].describe()
```

## Training

Now let's train a model on `train_data`. Here we create an instance of `TabularPredictor` with the argument `label` set to be the label column name. Then the `fit` method will automatically train the model. 

The training will take a few seconds. If you worry about taking too long, you could use the `time_limit` argument to set the maximal time AutoGluon can use for training. For example, `fit(train_data, time_limit=60)` will limit the time to be 1 minute.



```{.python .input}
predictor = TabularPredictor(label=label).fit(train_data)
```

You can find what AutoGluon tried by the log information. Roughly speaking, it does three things:
    
1. Identify it's a classification or a regression task based on the label values. 
1. Identify the feature column data types and convert them into proper features
1. Try various machine learning models with different hyper-parameters, and combine them together to form the final model

You can customize every step such as trying a different feature extraction method, machine learning model, or evaluation metric. We will cover them later. 

## Prediction

Last, load a separate test data to predict with the trained model.

```{.python .input}
test_data = TabularDataset('https://autogluon.s3.amazonaws.com/datasets/Inc/test.csv')
# delete the label column to prove we're not cheating
test_data_no_label = test_data.drop(columns=[label])
y_pred = predictor.predict(test_data_no_label)
y_pred.head()
```

You could use the `evaluate_predictions` method to test the prediction performance. Here we pass `slient=True` to disable AutoGluon to print log. 

```{.python .input}
predictor.evaluate_predictions(
    y_true=test_data[label], y_pred=y_pred, silent=True)
```

## Summary

Now you see how simply AutoGluon it is to predict a column on a tabular dataset. You first load the data via `TabularDataset`, and next call the `fit` method in the `TabularPredictor` to automatically train the model without specifying any other hyper-parameters. Then you can use the `predict` method to predict on new data. In most cases, you can stop read right now and turn to use AutoGluon to solve your own problems. 

If you want to know about AutoGluon, you can:

- A
- B
- C
