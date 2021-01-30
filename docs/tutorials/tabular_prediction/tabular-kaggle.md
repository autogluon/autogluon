# How to use AutoGluon for Kaggle competitions
:label:`sec_tabularkaggle`

This tutorial will teach you how to use AutoGluon to become a serious Kaggle competitor without writing lots of code.
We first outline the general steps to use AutoGluon in Kaggle contests. Here, we assume the competition involves tabular data which are stored in one (or more) CSV files.

1) Run Bash command: pip install kaggle

2) Navigate to: https://www.kaggle.com/account and create an account (if necessary).
Then , click on "Create New API Token" and move downloaded file to this location on your machine: `~/.kaggle/kaggle.json`. For troubleshooting, see [Kaggle API instructions](https://www.kaggle.com/docs/api).

3) To download data programmatically: Execute this Bash command in your terminal:

`kaggle competitions download -c [COMPETITION]`

Here, [COMPETITION] should be replaced by the name of the competition you wish to enter.
Alternatively, you can download data manually: Just navigate to website of the Kaggle competition you wish to enter, click "Download All", and accept the competition's terms.

4) If the competition's training data is comprised of multiple CSV files, use [pandas](https://pandas.pydata.org/pandas-docs/stable/user_guide/merging.html) to properly merge/join them into a single data table where rows = training examples, columns = features.

5) Run autogluon `fit()` on the resulting data table.

6) Load the test dataset from competition (again making the necessary merges/joins to ensure it is in the exact same format as the training data table), and then call autogluon `predict()`.  Subsequently use [pandas.read_csv](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html) to load the competition's `sample_submission.csv` file into a Dataframe, put the AutoGluon predictions in the right column of this Dataframe, and finally save it as a CSV file via [pandas.to_csv](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_csv.html). If the competition does not offer a sample submission file, you will need to create the submission file yourself by appropriately reformatting AutoGluon's test predictions.

7) Submit your predictions via Bash command:

`kaggle competitions submit -c [COMPETITION] -f [FILE] -m ["MESSAGE"]`

Here, [COMPETITION] again is the competition's name, [FILE] is the name of the CSV file you created with your predictions, and ["MESSAGE"] is a string message you want to record with this submitted entry. Alternatively, you can  manually upload your file of predictions on the competition website.

8) Finally, navigate to competition leaderboard website to see how well your submission performed!
It may take time for your submission to appear.



Below, we demonstrate how to do steps (4)-(6) in Python for a specific Kaggle competition: [ieee-fraud-detection](https://www.kaggle.com/c/ieee-fraud-detection/).
This means you'll need to run the above steps with `[COMPETITION]` replaced by `ieee-fraud-detection` in each command.  Here, we assume you've already completed steps (1)-(3) and the data CSV files are available on your computer. To begin step (4), we first load the competition's training data into Python:

```
import pandas as pd
import numpy as np
from autogluon.tabular import TabularPredictor

directory = '~/IEEEfraud/'  # directory where you have downloaded the data CSV files from the competition
label = 'isFraud'  # name of target variable to predict in this competition
eval_metric = 'roc_auc'  # Optional: specify that competition evaluation metric is AUC
save_path = directory + 'AutoGluonModels/'  # where to store trained models

train_identity = pd.read_csv(directory+'train_identity.csv')
train_transaction = pd.read_csv(directory+'train_transaction.csv')
```

Since the training data for this competition is comprised of multiple CSV files, we just first join them into a single large table (with rows = examples, columns = features) before applying AutoGluon:

```
train_data = pd.merge(train_transaction, train_identity, on='TransactionID', how='left')
```

Note that a left-join on the `TransactionID` key happened to be most appropriate for this Kaggle competition, but for others involving multiple training data files, you will likely need to use a different join strategy (always consider this very carefully). Now that all our training data resides within a single table, we can apply AutoGluon. Below, we specify the `presets` argument to maximize AutoGluon's predictive accuracy which usually requires that you run `fit()` with longer time limits (3600s below should likely be increased in your run):
```
predictor = TabularPredictor(label=label, eval_metric=eval_metric, path=save_path, verbosity=3).fit(
    train_data, presets='best_quality', time_limit=3600
)

results = predictor.fit_summary()
```

Now, we use the trained AutoGluon Predictor to make predictions on the competition's test data. It is imperative that multiple test data files are joined together in the exact same manner as the training data. Because this competition is evaluated based on the AUC (Area under the ROC curve) metric, we ask AutoGluon for predicted class-probabilities rather than class predictions. In general, when to use `predict` vs `predict_proba` will depend on the particular competition.

```
test_identity = pd.read_csv(directory+'test_identity.csv')
test_transaction = pd.read_csv(directory+'test_transaction.csv')
test_data = pd.merge(test_transaction, test_identity, on='TransactionID', how='left')  # same join applied to training files

y_predproba = predictor.predict_proba(test_data)
y_predproba.head(5)  # some example predicted fraud-probabilities
```

When submitting predicted probabilities for classification competitions, it is imperative these correspond to the same class expected by Kaggle. For binary classification tasks, you can see which class AutoGluon's predicted probabilities correspond to via:

```
predictor.positive_class
```

For multiclass classification tasks, you can see which classes AutoGluon's predicted probabilities correspond to via:

```
predictor.class_labels  # classes in this list correspond to columns of predict_proba() output
```

Alternatively, the following command should clarify which predicted-probability corresponds to which class:

```
y_predproba = predictor.predict_proba(test_data)
```

Now that we have made a prediction for each row in the test dataset, we can submit these predictions to Kaggle. Most Kaggle competitions provide a sample submission file, in which you can simply overwrite the sample predictions with your own as we do below:

```
submission = pd.read_csv(directory+'sample_submission.csv')
submission['isFraud'] = y_predproba
submission.head()
submission.to_csv(directory+'my_submission.csv', index=False)
```

We have now completed steps (4)-(6) from the top of this tutorial. To submit your predictions to Kaggle, you can run the following command in your terminal (from the appropriate directory):

`kaggle competitions submit -c ieee-fraud-detection -f sample_submission.csv -m "my first submission"`

You can now play with different `fit()` arguments and feature-engineering techniques to try and maximize the rank of your submissions in the Kaggle Leaderboard!


**Tips to maximize predictive performance:**

   - Be sure to specify the appropriate evaluation metric if one is specified on the competition website! If you are unsure which metric is best, then simply do not specify this argument when invoking `fit()`; AutoGluon should still produce high-quality models by automatically inferring which metric to use.

   - If the training examples are time-based and the competition test examples come from future data, we recommend you reserve the most recently-collected training examples as a separate validation dataset passed to `fit()`. Otherwise, you do not need to specify a validation set yourself and AutoGluon will automatically partition the competition training data into its own training/validation sets.

   - Beyond simply specifying `presets = 'best_quality'`, you may play with more advanced `fit()` arguments such as: `num_bag_folds`, `num_stack_levels`, `num_bag_sets`, `hyperparameter_tune_kwargs`, `hyperparameters`, `refit_full`. However we recommend spending most of your time on [feature-engineering](https://www.coursera.org/lecture/competitive-data-science/overview-1Nh5Q) and just specifying `presets = 'best_quality'` inside the call to `fit()`.


**Troubleshooting:**

- Check that you have the right user-permissions on your computer to access the data files downloaded from Kaggle.

- For issues downloading Kaggle data or submitting predictions, check your Kaggle account setup and the [Kaggle FAQ](https://www.kaggle.com/general/14438).
