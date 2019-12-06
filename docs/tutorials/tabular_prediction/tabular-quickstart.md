# Predicting Columns in a Table - Quick Start
:label:`sec_tabularquick`

Via a simple `fit()` call, AutoGluon can produce highly-accurate models to predict the values in one column of data table based on the rest of the columns' values (both classification and regression).
To see this in action, let's load first load training data from a CSV file into an AutoGluon Dataset object. This object is essentially equivalent to a [Pandas DataFrame](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html) and all the same methods can be applied to both. 

```{.python .input}
import autogluon as ag
from autogluon import TabularPrediction as task

train_data = task.Dataset(file_path='https://autogluon.s3-us-west-2.amazonaws.com/datasets/AdultIncomeBinaryClassification/train_data.csv') # can be local CSV file as well, returns pd.Dataframe-like object.
train_data = train_data.head(500) # subsample 500 data points for faster demo
print(train_data.head())
```

Note that we loaded data from a CSV file stored in the cloud (AWS s3 bucket), but you can you specify a local file-path instead if you have already downloaded the CSV file to your own machine (eg. using `wget`).
Each row in the table `train_data` corresponds to a single training example. In this particular dataset, each row corresponds to an individual person, and the columns contain various features reported during a census. 

Let's first try to use these features to predict whether the person's income exceeds 50K or not, which is recorded in the `class` column of this table.

```{.python .input}
label_column = 'class'
print(train_data[label_column].describe())
```

Now use AutoGluon to train some models:

```{.python .input}
savedir = 'agModels-predictClass' # specifies folder where to store trained models
time_limits = 60 # train various models for about 60sec

predictor = task.fit(train_data=train_data, label=label_column, output_directory=savedir, time_limits=time_limits)
```

Next, load some separate test data to demonstrate how to make predictions on new examples at inference time:

```{.python .input}
test_data = task.Dataset(file_path='https://autogluon.s3-us-west-2.amazonaws.com/datasets/AdultIncomeBinaryClassification/test_data.csv')
y_test = test_data[label_column]
test_data = test_data.drop(labels=[label_column],axis=1) # delete label column to prove we are not cheating
print(test_data.head())
```

We use our trained models to make predictions on the new data and then evaluate performance: 

```{.python .input}
predictor = task.load(savedir) # unnecessary, just demonstrates how to load previously-trained predictor from file

y_pred = predictor.predict(test_data)
print("Predictions:  ", y_pred)
perf = predictor.evaluate(y_true=y_test, y_pred=y_pred, auxiliary_metrics=True)
```

Now you're ready to try AutoGluon on your own tabular datasets!   
As long as they're stored in a popular format like CSV, you should be able to achieve strong predictive performance with just 2 lines of code:
```
from autogluon import TabularPrediction as task
predictor = task.fit(train_data=task.Dataset(file_path=<file-name>), label_column=<variable-name>)
```


## Description of fit():

Here we provide more details about what happened during `fit()`. 
Since there are only two possible values of the `class` variable, this was a binary classification problem, for which an appropriate performance metric is *accuracy*.
AutoGluon automatically infers all this as well as the type of each feature (ie. which columns contain continuous numbers vs. discrete categories), and can also automatically handle common issues like missing data and rescaling feature values.


As we did not specify separate validation data, AutoGluon automatically choses a random training/validation split of the data. 
Rather than just a single model, AutoGluon trains many models and ensembles them together to ensure superior predictive performance. 
By default, AutoGluon tries to fit various types of models including neural networks and tree ensembles.
Each type of model has various hyperparameters, which the user would normally need to manually specify, but AutoGluon automatically finds values of these hyperparameters which produce the best performance on the validation data. This involves repeatedly training models under different hyperparameter settings and evaluating their performance, which can get computationally-intensive, so `fit()` can parallelize this process across multiple threads (and machines if distributed resources are available). To control runtimes, you can specify various arguments in fit(): `time_limits` which stops training new models after the specified amount of time (sec) has passed, `num_trials` which specifies how many hyperparameter configurations to try for each type of model. You can also make an individual training run of each model quicker by specifying appropriate arguments as demonstrated in the subsequent **In-Depth** tutorial.


For tabular problems, AutoGluon stores various information produced during `fit()` in a `Trainer` object.
We can use this object to view the validation accuracy of each individual model that was trained during `fit()`, as well as various model ensembles AutoGluon considered after the models were trained.

```{.python .input}
trainer = predictor.load_trainer()
print("AutoGluon trained %s models during fit(). Their validation performance is below:\n" % len(trainer.model_names))

print(trainer.model_performance)
```

We can also see what things AutoGluon inferred about our prediction task:

```{.python .input}
print("AutoGluon infers problem type is: ", trainer.problem_type)
print("AutoGluon categorized the features as: ", trainer.feature_types_metadata)
```

We can also view extremely detailed information about the hyperparameter optimization process, including what was the best hyperparameter configuration for each type of model and how well each hyperparameter configuration performed on the validation data:

```
import pprint  # will print tons of information so command isn't executed here:
pprint.PrettyPrinter(indent=2).pprint(trainer.hpo_results)
```

To demonstrate that `fit()` can also automatically handle regression tasks, we now try to predict the `age` variable in the table based on the other features.
Here, we run a quicker `fit()` that simply trains one model of each type without any hyperparameter optimization.

```{.python .input}
label_column = 'age'
print(train_data[label_column].describe())

predictor_educ = task.fit(train_data=train_data, output_directory="agModels-predictEducation", 
                          label=label_column, hyperparameter_tune=False)
```

Note that we didn't need to tell AutoGluon this is a regression problem, it automatically inferred this from the data.

Finally, we need to shutdown AutoGluon's remote workers which `fit()` uses to train multiple models simultaneously in multi-thread / distributed settings. 
