# AutoGluon.Shift Tutorial

## Introduction

Distributional shift is when there is a difference between the training and test data in a prediction problem.  In this tutorial we introduce the `C2STShiftDetector` class which will detect and explain a change in the covariate (X) distributions, a phenomenon that we call XShift.  This is one of the ways in which distributional shift can manifest, but not the only one.


```{.python .input}
import autogluon.shift as sft
```


```{.python .input}
from helpers import load_adult_data, sim_cov_shift
from sklearn import metrics
from autogluon.tabular import TabularPredictor
import plotnine as p9
import bisect
```

## Importing data

We will import the adult dataset.  In the following analysis we will construct a dataset with covariate shift.  This means that we will need to identify a feature that can be used to bias the training sample, in order to make it not representative of the test population.  As we can see the marital status has a good mix of married and never-married, making it a potential candidate 


```{.python .input}
train, test = load_adult_data()

train.head()
```


```{.python .input}
train[['marital-status']].value_counts(normalize=True)
```

## Detecting XShift

First we will apply the XShift detector to the original adults dataset.  This detector uses the Classifier 2 Sample Test, hence it is `C2STShiftDetector`.  We see that our test does not detect a substantial difference between the training and test X distributions.  This was determined by calculating the balanced accuracy (50.02%) for a classifier, which predicts if a sample is in the test or training set.  This is so close to 50% (random guessing) that we suspect that the adults training/test sets are a random sample split.


```{.python .input}
xsd = sft.C2STShiftDetector(TabularPredictor,
                            label='class',
                            classifier_kwargs={'verbosity': 0,
                                              'path': 'AutogluonModels'})
```


```{.python .input}
xsd.fit(train, test)
```


```{.python .input}
sumry = xsd.summary()

print(sumry)
```


```{.python .input}
xsd.C2ST.test_stat
```

## Simulating covariate shift

In this section, we will simulate covariate shift for the adults dataset.  We do this by finding a variable that has both high enough entropy to be useful to bias the training data, but also has some bearing on the penultimate prediction.  We find that marital status is one such variable, and the function `sim_cov_shift` creates a biased sample based on this.


```{.python .input}
pred = TabularPredictor(label='class', 
                        verbosity=0, 
                        problem_type='binary',
                        path='AutogluonModels').fit(train)
```


```{.python .input}
yhat = pred.predict(test)
metrics.balanced_accuracy_score(yhat, test['class'])
```


```{.python .input}
pred.feature_importance(test)
```


```{.python .input}
train_cs, test_cs = sim_cov_shift(train, test)
```

We can see that the new training data underrepresents the 'Married-civ-spouse' status while the test data overrepresents it.


```{.python .input}
train_cs.value_counts('marital-status',normalize=True)
```


```{.python .input}
test_cs.value_counts('marital-status',normalize=True)
```

We now train the XShift detector on the shifted data.


```{.python .input}
xsd = sft.C2STShiftDetector(TabularPredictor,
                            label='class',
                            classifier_kwargs={'verbosity': 0,
                                              'path': 'AutogluonModels'})
```


```{.python .input}
xsd.fit(train_cs, test_cs)
```

You can print a detailed summary of the results.  In the case of a detection you can see what the feature importances are, which basically tells you what variables are shifted the most between training and test.


```{.python .input}
sumry = xsd.summary()

print(sumry)
```

You can also obtain anomaly scores for the test samples.  These are the estimated probability that the test samples are in the test set (based on two-fold cross-validation).  Basically, it is a measure of how much a test sample looks like the rest of the test set versus the training set.


```{.python .input}
ano_data = xsd.anomaly_scores()
```


```{.python .input}
ano_test = ano_data.join(test_cs)
ano_test.head()
```
