# Overview
FairPredictor is a postprocessing approach for enforcing fairness, with support for a wide range of performance metrics and fairness criteria, and support for inferred attributes, i.e. it does not require access to protected attributes at test time. 
Under the hood, FairPredictor works by adjusting the decision boundary for each group individually. Where groups are not available, it makes use of inferred group membership to adjust decision boundaries.

The key idea underlying this toolkit is that for a wide range of use cases, the most suitable classifier should do more than maximize some form of accuracy.
We offer a general toolkit that allows different measures to be optimized and additional constraints to be imposed by tuning the behavior of a binary predictor on validation data.

For example, classifiers can be tuned to maximize performance for a wide range of metrics such as:

* Accuracy
* Balanced Accuracy
* F1 score
* MCC
* Custom utility functions

While also approximately satisfying a wide range of group constraints such as:

* Demographic Parity (The idea that positive decisions should occur at the same rates for all protected groups, for example for men at the same rate as for women)
* Equal Opportunity (The recall should be the same for all protected groups)
* Minimum recall constraints (The recall should be above a particular level for all groups)
* Minimum Precision constraints (The precision should be above a particular level for all groups)
* Custom Fairness Metrics

The full set of constraints and objectives can be seen in the list of measures at the bottom of the document. 

## Example usage

    # Load and train a baseline classifier

    from autogluon.tabular import TabularDataset, TabularPredictor
    from autogluon.fair import FairPredictor
    from autogluon.fair.utils import group_metrics as gm
    train_data = TabularDataset('https://autogluon.s3.amazonaws.com/datasets/Inc/train.csv')
    test_data = TabularDataset('https://autogluon.s3.amazonaws.com/datasets/Inc/test.csv')
    predictor = TabularPredictor(label='class').fit(train_data=train_data)
    
    # Modify predictor to enforce fairness over the train_data with respect to groups given by the column 'sex'
    fpredictor = FairPredictor(predictor,train_data,'sex')
    # Maximize accuracy while enforcing that the demographic parity (the difference in positive decision rates between men and women is at most 0.02)
    fpredictor.fit(gm.accuracy,gm.demographic_parity,0.02)
    
    # Evaluate on test data
    fpredictor.predict(test_data)
    
    # Evaluate a range of performance measures, and compare against original classifier on test data
    fpredictor.evaluate(test_data, verbose= True)

|                   |   original |   updated |
|:------------------|-----------:|----------:|
| Accuracy          |   0.876446 |  0.853926 |
| Balanced Accuracy |   0.796708 |  0.757129 |
| F1 score          |   0.712414 |  0.650502 |
| MCC               |   0.640503 |  0.568616 |
| Precision         |   0.795636 |  0.752408 |
| Recall            |   0.644953 |  0.572908 |
| roc_auc           |   0.931573 |  0.827535 |

    # Evaluate against a range of standard fairness definitions and compare against original classifier on test data
    fpredictor.evaluate_fairness(test_data, verbose= True)

|                                                         |   original |    updated |
|:--------------------------------------------------------|-----------:|-----------:|
| Class Imbalance                                         |  0.195913  | 0.195913   |
| Demographic Parity                                      |  0.166669  | 0.00744171 |
| Disparate Impact                                        |  0.329182  | 0.959369   |
| Maximal Group Difference in Accuracy                    |  0.0936757 | 0.0684973  |
| Maximal Group Difference in Recall                      |  0.0590432 | 0.326703   |
| Maximal Group Difference in Conditional Acceptance Rate |  0.0917708 | 1.04471    |
| Maximal Group Difference in Acceptance Rate             |  0.0174675 | 0.347018   |
| Maximal Group Difference in Specificity                 |  0.0518869 | 0.0594707  |
| Maximal Group Difference in Conditional Rejectance Rate |  0.0450807 | 0.229982   |
| Maximal Group Difference in Rejection Rate              |  0.0922794 | 0.157476   |
| Treatment Equality                                      |  0.0653538 | 5.07559    |
| Generalized Entropy                                     |  0.0666204 | 0.080265   |

    # Evaluate a range of performance measures per group, and compare against original classifier on test data
    fpredictor.evaluate_groups(test_data, verbose= True, return_original=True)

|                                    |   Accuracy |   Balanced Accuracy |   F1 score |       MCC |   Precision |    Recall |   roc_auc |   Positive Count |   Negative Count |   Positive Label Rate |   Positive Prediction Rate |
|:-----------------------------------|-----------:|--------------------:|-----------:|----------:|------------:|----------:|----------:|-----------------:|-----------------:|----------------------:|---------------------------:|
| ('original', 'Overall')            |  0.876446  |          0.796708   | 0.712414   | 0.640503  |   0.795636  | 0.644953  | 0.931573  |             2318 |             7451 |              0.237281 |                 0.192343   |
| ('original', ' Female')            |  0.938583  |          0.787403   | 0.675241   | 0.649242  |   0.780669  | 0.594901  | 0.949251  |              353 |             2936 |              0.107327 |                 0.0817878  |
| ('original', ' Male')              |  0.844907  |          0.790981   | 0.718881   | 0.619052  |   0.798137  | 0.653944  | 0.91321   |             1965 |             4515 |              0.303241 |                 0.248457   |
| ('original', 'Maximum difference') |  0.0936757 |          0.00357813 | 0.04364    | 0.03019   |   0.0174675 | 0.0590432 | 0.0360405 |             1612 |             1579 |              0.195913 |                 0.166669   |
| ('updated', 'Overall')             |  0.853926  |          0.757129   | 0.650502   | 0.568616  |   0.752408  | 0.572908  | 0.827535  |             2318 |             7451 |              0.237281 |                 0.180674   |
| ('updated', ' Female')             |  0.899362  |          0.877586   | 0.644468   | 0.614161  |   0.519031  | 0.849858  | 0.949251  |              353 |             2936 |              0.107327 |                 0.175737   |
| ('updated', ' Male')               |  0.830864  |          0.74397    | 0.652284   | 0.579829  |   0.866049  | 0.523155  | 0.91321   |             1965 |             4515 |              0.303241 |                 0.183179   |
| ('updated', 'Maximum difference')  |  0.0684973 |          0.133616   | 0.00781595 | 0.0343327 |   0.347018  | 0.326703  | 0.0360405 |             1612 |             1579 |              0.195913 |                 0.00744171 |

## Why Another Fairness Library?

Fundamentally, most existing fairness methods are not appropriate for ensemble methods like AutoGluon.
Under the hood, autogluon makes use of many different types of classifiers trained on random subsets of the data, and it's inherent complexity makes fairness methods that iteratively retrain classifiers punitively slow. The use of random subsets makes AutoGluon robust to small amounts of mislabeled data, but also means that methods that iteratively make small changes to the training data to enforce fairness can have unpredictable behavior. At same time, the many different types of sub-classifiers used mean that any method inprocessing method that requires the alteration of every method used to train a sub-classifier is not feasible.

That said, we make several design decisions which we believe make for a better experience for data scientists:

### Fine-grained control of behavior

#### Wide Choice of performance measure

Unlike other approaches to fairness, FairPredictor allows the optimization of arbitrary performance measures such as F1 or MCC, subject to fairness constraints. This can substantially decrease the fairness/performance trade-off with, for example, F1 scores being 3-4% higher when directly optimized for rather than accuracy.

#### Wide Choice of Fairness Measures

Rather than offering a range of different fairness methods that enforce a small number of fairness definitions through a variety of different methods, we offer one method that can enforce a much wider range of fairness definitions out of the box, alongside support for custom fairness definitions.

Of the set of metrics discussed in [Verma and Rubin](https://fairware.cs.umass.edu/papers/Verma.pdf), and the metrics measured by [Sagemaker Clarify](https://docs.aws.amazon.com/sagemaker/latest/dg/clarify-measure-post-training-bias.html), out of the box FairPredictor offers the ability to both measure and enforce 7 of the 8 metrics used to evaluate classifier decision measured in Verma and Rubin, and 11 of the 13 measures used in Clarify.

#### Direct Remedy of Harms

Many fairness measures can be understood as identifying a harm, and then equalizing this harm across the population as a whole. For example, the use demographic parity of identifies people as being harmed by a low selection rate, which is then set to be the same for all groups, while equal opportunity identifies people as being harmed by low recall.

As an alternative to equalizing the harm across the population, we allow data scientists to specify minimum rates of e.g., recall, precision, or selection rate for every group, with one line of code. E.g. 

    fpredictor.fit(gm.accuracy, gm.precision.min, 0.5)

 force the found classifier to have a precision of at least 0.5 for every group.

These constraints have wider uses outside of fairness. For example, a classifier can be trained to identify common defects across multiple factory lines, and its behavior can be altered to enforce a recall of 90% of all defects for each line (at a cost of increased false positives).

    fpredictor.fit(gm.accuracy, gm.recall.min, 0.9)

#### Support for Utility based approaches

We provide support for the utility based approach set out in [Fairness On The Ground: Applying Algorithmic Fairness Approaches To Production Systems](https://arxiv.org/pdf/2103.06172.pdf), whereby different thresholds can be selected per group to optimize a utility-based objective.

Utility functions can be defined in one line.

For example, if we have a situation where an ML system identifies potential problems that require intervening, it might be that every intervention has a cost of 1, regardless of if it was needed, but a missed intervention that was needed has a cost of 5. Finally, not making an intervention when one was not needed has a cost of 0. This can be written as:

    my_utility=gm.Utility([1, 1, 5, 0], 'Testing Costs')

and optimized alongside fairness or performance constraints. For example,

    fpredictor.fit(my_utility)

optimizes the utility, while

    fpredictor.fit(my_utility, gm.accuracy, 0.5)

optimizes the utility subject to the requirement that the classifier accuracy can not drop below 0.5.
  
#### Support for user-specified performance and fairness measures

As well as providing support for enforcing a wide range of performance and fairness measures, we allow users to define their own metrics and fairness measures.

For example, a custom implementation of recall can be defined as:

    my_recall = gm.GroupMetric(lambda TP, FP, FN, TN: (TP) / (TP + FN), 'Recall')

and then the maximum difference in recall between groups (corresponding to the fairness definition of Equal Opportunity) is provided by calling `my_recall.diff`, and the minimum recall over any group (which can be used to ensure that the recall is above a particular value for every group) is given by `my_recall.min`. 

## Altering Behavior

The behavior of classifiers can be altered through the fit function.
Given a pretrained binary predictor, we define a fair classifier that will allow us to alter the existing behavior on a validation dataset, by using a labeled attribute 'sex'.

    fpredictor = FairPredictor(predictor,  validation_data, 'sex')

the fit function takes three arguments that describe an objective such as accuracy or F1 that should be optimized, and a constraint such as the demographic parity violation should be below 2%.

This takes the form:

    fpredictor.fit(gm.f1, gm.demographic_parity, 0.02)

The constraint and objective can be swapped so the code:

    fpredictor.fit(gm.demographic_parity, gm.f1, 0.75)

will find the method with the lowest demographic parity violation such that F1 is above 0.75.
If, for example, you wish to optimize F1 without any additional constraints, you can just call:

    fpredictor.fit(gm.f1)

Note that the default behavior (we should minimize the demographic parity violation, but maximize F1) is inferred from standard usage but can be overwritten by setting the optional parameters `greater_is_better_obj` and `greater_is_better_const` to `True` or `False`.

Where constraints cannot be satisfied, for example, if we require that the F1 score must be above 1.1, `fit` returns the solution closest to satisfying it.

## Measuring Behavior

A key challenge in deciding how to alter the behavior of classifiers is that these decisions have knock-on effects elsewhere. For example, increasing the precision of classifiers, will often decrease their recall and vice versa.

In the same way, many fairness definitions may be at odds with one another, and increasing the fairness with respect to one definition can decrease it with respect to other definitions.

As such, we offer a range of methods for evaluating the performance of classifiers.

    fpredictor.evaluate(data (optional), groups (optional), dictionary_of_methods (optional), verbose=False)

By default, this method reports the standard binary evaluation criteria of autogluon for both the original and updated predictor, over the data used by fit. The behavior can be altered by providing either alternate data or a new dictionary of methods. Where groups are not provided, it will use the same groups as passed to `fit`, but this can be altered. If verbose is set to true, the table contains the long names of methods, otherwise it reports the dictionary keys.

    fpredictor.evaluate_fairness(data (optional), groups (optional), dictionary_of_methods (optional), verbose=False)

By default, this method reports the standard fairness metrics of SageMaker Clarify for both the original and updated predictor, over the data used by fit. The behavior can be altered by providing either alternate data or a new dictionary of methods. Where groups is not provided, it will use the same groups as passed to `fit`, but this can be altered. If verbose is set to true, the table contains the long names of methods, otherwise it reports the dictionary keys.

    fpredictor.evaluate_groups(data (optional), groups (optional), dictionary_of_methods (optional), return_original=False, verbose=False)

By default, this method reports, per group, the standard binary evaluation criteria of autogluon for both the updated predictor only, over the data used by fit. The behavior can be altered by providing either alternate data or a new dictionary of methods. Where groups is not provided, it will use the same groups as passed to `fit`, but this can be altered. If you wish to also see the per group performance of the original classifier, use `return_original=True` to receive a dict containing the per_group performance of the original and updated classifier. If verbose is set to true, the table contains the long names of methods, otherwise it reports the dictionary keys.

## Fairness using Inferred Attributes
  
In many cases, the attribute you wish to be fair with respect to such as `sex` may not be available at test time. In this case you can make use of inferred attributes predicted by another classifier. This can be done by defining the fair predictor in the following way.

    fpredictor = fair.FairPredictor(predictor,  data, 'sex', inferred_groups=attribute_pred)

where `attribute_pred` is another autogluon predictor trained to predict the attribute, such as `sex` you wish to infer.
Then `fpredictor` can be used in same way described above.

Note that the labeled attribute is used to evaluate fairness, and the use of the inferred attributes are tuned to optimize fairness with respect to the labeled attributes. This means that even if the inferred attributes are not that accurate, they can be still used to enforce fairness, albeit with a drop in performance.

To make it easier to use inferred attributes, we provide a helper function:

    predictor, attribute_pred = fair.learners.inferred_attribute_builder(train_data, 'class', 'sex')

This allows for the easy training of two tabular classifiers, one called `predictor` to predict the target label `class` without using the attribute `sex` and one called `attribute_pred` to predict `sex` without using the the target label.

## Fairness on COMPAS using Inferred Attributes

We demonstrate how to enforce a wide range of fairness definitions on the COMPAS dataset. This dataset records paroles caught violating the terms of parole. As it measures who was caught, it is strongly influenced by policing and environmental biases, and should not be confused with a measurement of who actually violated their terms of parole. See [this paper](https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/file/92cc227532d17e56e07902b254dfad10-Paper-round1.pdf) for a discussion of its limitations and caveats. 
We use it because it is a standard fairness dataset that captures such strong differences in outcome between people identified as African-American and everyone else, that classifiers trained on this dataset violate most definitions of fairness.

As many of the ethnic groups are too small for reliable statistical estimation, we only consider differences is in outcomes between African-Americans vs. everyone else (labeled as other).
We load and preprocess the COMPAS dataset, splitting it into three roughly equal partitions of train, validation, and test:

    import pandas as pd
    import   numpy   as   np
    from autogluon.fair import FairPredictor, inferred_attribute_builder
    from autogluon.fair.utils import group_metrics as gm
    all_data = pd.read_csv('https://github.com/propublica/compas-analysis/raw/master/compas-scores-two-years.csv')
    condensed_data=all_data[['sex','race','age', 'juv_fel_count', 'juv_misd_count', 'juv_other_count', 'priors_count', 'age_cat', 'c_charge_degree','two_year_recid']].copy()
    condensed_data.replace({'Caucasian':'Other', 'Hispanic':'Other', 'Native American':'Other', 'Asian':'Other'},inplace=True)
    train=condensed_data.sample(frac=0.3, random_state=0)
    val_and_test=condensed_data.drop(train.index)
    val=val_and_test.sample(frac=0.5, random_state=0)
    test=val_and_test.drop(val.index)

To enforce fairness constraints without access to protected attributes at test time, we train two classifiers to infer the 2-year recidivism rate, and ethnicity.
  
    predictor2, protected = inferred_attribute_builder(train, 'two_year_recid', 'race')

From these a single predictor that maximizes accuracy while reducing the demographic parity violation to less than 2.5% can be trained by running:

    fpredictor=FairPredictor(predictor2, val, 'race', protected)
    fpredictor.fit(gm.accuracy, gm.demographic_parity, 0.025)

However, instead we will show how a family of fairness measures can be individually optimized. First, we consider the measures of Sagemaker Clarify that we support. The following code plots a table showing the change in accuracy and the fairness measure on a held-out test set as we decrease the fairness measure to less than 0.025 (on validation) for all measures except for disparate impact which we raise to above 0.975.
We define a helper function for evaluation:

    def evaluate(fpredictor,use_metrics):
        "Print a table showing the accuracy drop that comes with enforcing fairness"
        extra_metrics= {**use_metrics, 'accuracy':gm.accuracy}
        collect=pd.DataFrame(columns=['Measure (original)', 'Measure (updated)', 'Accuracy (original)', 'Accuracy (updated)'])
        for d in use_metrics.items():
            if d[1].greater_is_better is False:
                fpredictor.fit(gm.accuracy,d[1],0.025)
            else:
                fpredictor.fit(gm.accuracy,d[1],1-0.025)
            tmp=fpredictor.evaluate_fairness(test,metrics=extra_metrics)
            collect.loc[d[1].name]=np.concatenate((np.asarray(tmp.loc[d[0]]),np.asarray(tmp.loc['accuracy'])),0)
        print(collect.to_markdown(())

We can now contrast the behavior of a fair classifier that relies on access to the protected attribute at test time with one that infers it.

    # we first create a classifier using the protected attribute
    predictor=TabularPredictor(label='two_year_recid').fit(train_data=train)
    
    fpredictor = FairPredictor(predictor, val, 'race', )
    
    evaluate(fpredictor, gm.clarify_metrics)

This returns the following table, which shows little drop in accuracy compared to the original and in some cases, even an improvement. N.B. Class Imbalance is a property of the dataset and cannot be updated.

|                                                         |   Measure (original) |   Measure (updated) |   Accuracy (original) |   Accuracy (updated) |
|:--------------------------------------------------------|---------------------:|--------------------:|----------------------:|---------------------:|
| Class Imbalance                                         |            0.132203  |          0.132203   |              0.666139 |             0.663762 |
| Demographic Parity                                      |            0.283466  |          0.0328274  |              0.666139 |             0.659406 |
| Disparate Impact                                        |            0.514436  |          0.951021   |              0.666139 |             0.677228 |
| Maximal Group Difference in Accuracy                    |            0.0469919 |          0.0532531  |              0.666139 |             0.663762 |
| Maximal Group Difference in Recall                      |            0.236378  |          0.00533019 |              0.666139 |             0.67604  |
| Maximal Group Difference in Conditional Acceptance Rate |            0.380171  |          0.0555107  |              0.666139 |             0.670099 |
| Maximal Group Difference in Acceptance Rate             |            0.0202594 |          0.0438892  |              0.666139 |             0.658614 |
| Maximal Group Difference in Specificity                 |            0.251729  |          0.0831756  |              0.666139 |             0.664158 |
| Maximal Group Difference in Conditional Rejectance Rate |            0.29054   |          0.0107142  |              0.666139 |             0.670891 |
| Maximal Group Difference in Rejection Rate              |            0.0620499 |          0.0743351  |              0.666139 |             0.663762 |
| Treatment Equality                                      |            0.933566  |          0.159398   |              0.666139 |             0.66099  |
| Generalized Entropy                                     |            0.16627   |          0.0508368  |              0.666139 |             0.442376 |

In contrast, even though the base classifiers have similar accuracy, when using inferred attributes (N.B. the base classifier is not directly trained to maximize accuracy, which is why it can have higher accuracy when it doesn't use race), we see a much greater drop in accuracy as fairness is enforced which is consistent with [Lipton et al.](https://arxiv.org/pdf/1711.07076.pdf)

    # Now using the inferred attributes
    
    fpredictor2 = FairPredictor(predictor2, val, 'race', protected)
    
    evaluate(fpredictor2,gm.clarify_metrics)

|                                                         |   Measure (original) |   Measure (updated) |   Accuracy (original) |   Accuracy (updated) |
|:--------------------------------------------------------|---------------------:|--------------------:|----------------------:|---------------------:|
| Class Imbalance                                         |            0.132203  |          0.132203   |              0.672871 |             0.666535 |
| Demographic Parity                                      |            0.21792   |          0.0344565  |              0.672871 |             0.584158 |
| Disparate Impact                                        |            0.512905  |          0.863017   |              0.672871 |             0.563564 |
| Maximal Group Difference in Accuracy                    |            0.0147268 |          0.00976726 |              0.672871 |             0.666535 |
| Maximal Group Difference in Recall                      |            0.231539  |          0.121319   |              0.672871 |             0.583762 |
| Maximal Group Difference in Conditional Acceptance Rate |            0.500941  |          0.00282887 |              0.672871 |             0.601188 |
| Maximal Group Difference in Acceptance Rate             |            0.0723272 |          0.145199   |              0.672871 |             0.585347 |
| Maximal Group Difference in Specificity                 |            0.139306  |          0.0397364  |              0.672871 |             0.593663 |
| Maximal Group Difference in Conditional Rejectance Rate |            0.080529  |          0.00827387 |              0.672871 |             0.662574 |
| Maximal Group Difference in Rejection Rate              |            0.0548552 |          0.0556917  |              0.672871 |             0.666535 |
| Treatment Equality                                      |            0.32195   |          0.0277123  |              0.672871 |             0.590099 |
| Generalized Entropy                                     |            0.196436  |          0.0508368  |              0.672871 |             0.442376 |

Similar results can be obtained using the metrics of Verma and Rubin, by running 

    evaluate(fpredictor, gm.verma_metrics)

|                                                 |   Measure (original) |   Measure (updated) |   Accuracy (original) |   Accuracy (updated) |
|:------------------------------------------------|---------------------:|--------------------:|----------------------:|---------------------:|
| Statistical Parity                              |            0.283466  |           0.0328274 |              0.666139 |             0.659406 |
| Predictive Parity                               |            0.0202594 |           0.0438892 |              0.666139 |             0.658614 |
| Maximal Group Difference in False Positive Rate |            0.251729  |           0.0775969 |              0.666139 |             0.667723 |
| Maximal Group Difference in False Negative Rate |            0.236378  |           0.0421043 |              0.666139 |             0.674455 |
| Equalized Odds                                  |            0.244053  |           0.0106539 |              0.666139 |             0.673663 |
| Conditional Use Accuracy                        |            0.0411546 |           0.0468682 |              0.666139 |             0.668119 |
| Predictive Equality                             |            0.236378  |           0.0421043 |              0.666139 |             0.674455 |
| Maximal Group Difference in Accuracy            |            0.0469919 |           0.0532531 |              0.666139 |             0.663762 |
| Treatment Equality                              |            0.933566  |           0.159398  |              0.666139 |             0.66099  |

and 

    evaluate(fpredictor2, gm.verma_metrics)
|                                                 |   Measure (original) |   Measure (updated) |   Accuracy (original) |   Accuracy (updated) |
|:------------------------------------------------|---------------------:|--------------------:|----------------------:|---------------------:|
| Statistical Parity                              |            0.21792   |          0.0344565  |              0.672871 |             0.584158 |
| Predictive Parity                               |            0.0723272 |          0.145199   |              0.672871 |             0.585347 |
| Maximal Group Difference in False Positive Rate |            0.139306  |          0.0397364  |              0.672871 |             0.593663 |
| Maximal Group Difference in False Negative Rate |            0.231539  |          0.108081   |              0.672871 |             0.582574 |
| Equalized Odds                                  |            0.185422  |          0.0309648  |              0.672871 |             0.586535 |
| Conditional Use Accuracy                        |            0.0635912 |          0.0632338  |              0.672871 |             0.627723 |
| Predictive Equality                             |            0.231539  |          0.108081   |              0.672871 |             0.582574 |
| Maximal Group Difference in Accuracy            |            0.0147268 |          0.00976726 |              0.672871 |             0.666535 |
| Treatment Equality                              |            0.32195   |          0.0277123  |              0.672871 |             0.590099 |

## Best Practices

It is common for machine learning algorithms to overfit training data. Therefore, if you want your fairness constraints to carry over to unseen data we recommend that they are enforced on a large validation set, rather than the training set. For low-dimensional datasets, Autogluon predictors are robust to overfitting and fairness constraints enforced on training data carry over to unseen test data. In fact, given the choice between enforcing fairness constraints on a large training set, vs. using a significantly smaller validation set, reusing the training set will often result in better generalization of the desired behavior to unseen data. However, this behavior is not guaranteed, and should always be empirically validated.

### Challenges with unbalanced data.

Many datasets are unbalanced both in the size of protected groups and in the prevalence of positive or negatively labeled data. When a rare group rarely receives positives outcomes, large datasets are needed to correctly estimate the rate of failure per group on positive data. This can make it very hard to reliably enforce or evaluate measures such as equal opportunity or minimum recall on unbalanced datasets, particularly where the baseline classifier has relatively high accuracy. The size and nature of the dataset needs to be carefully considered when choosing a fairness metric.

For example, on the historic dataset 'adult'; African Americans, despite being the second largest ethnicity after Caucasian, only make up around 10% of the dataset, with only 10% of them earning over 50k (N.B. adult is based on census data from 1994, in the same dataset, around 20% of white people earned over 50k). If an algorithm had a 10% error rate on this subset of the data, we are concerned with the behavior of around 10%^3 i.e., 0.1% of the data. This problem becomes even greater when looking at less prevalent groups.

For this reason, reliably guaranteeing high-accuracy across all groups, or that fairness measures are satisfied, requires access to rebalanced datasets, or much larger datasets than are needed for guaranteeing accuracy at the population level.

The file `../autogluon/examples/fair/sample_use.ipynb` has an example on the adult dataset where demographic parity is only weakly enforced on test data for the smaller groups `American-Indian-Eskimo`, and `Asian-Pacific-Islander` due to limited sample size.

## List of Measures

The remainder of the document lists the standard measures provided by the group_metrics library, which is imported as:

    from autogluon.fair.utils import group_metrics as gm
### Basic Structure

The majority of measures are defined as GroupMetrics or sub-objects of GroupMetrics.

A group measure is specified by a function that takes the number of True Positives, False Positives, False Negatives, and True Negatives and returns a score; A string specifying the name of the of the measure; and optionally a bool indicating if greater values are better than smaller ones. For example, accuracy is defined as:

    accuracy = gm.GroupMetric(lambda TP, FP, FN, TN: (TP + TN) / (TP + FP + FN + TN), 'Accuracy')

For efficiency, our approach relies on broadcast semantics and all operations in the function must be applicable to numpy arrays.

Having defined a GroupMetric it can be called in two ways. Either:

    accuracy(target_labels, predictions, groups)

Here target_labels and predictions are binary vectors corresponding to either the target ground-truth values, or the predictions made by a classifier, with 1 representing the positive label and 0 otherwise. Groups is simply a vector of values where each unique value is assumed to correspond to a distinct group.

The other way it can be called is by passing it a single 3D array of dimension 4 by number of groups by k, where k is the number of candidate classifiers that the measure should be computed over.

As a convenience, GroupMetrics automatically implements a range of functionality as sub-objects.

Having defined a metric as above, we have a range of different objects:

 *   `metric.av_diff` reports the average absolute difference of the method between groups.
 *   `metric.average` reports the average of the method taken over all groups.
 *   `metric.diff` reports the maximum difference of the method between any pair of groups.
 *   `metric.max` reports the maximum value for any group.
 *   `metric.min` reports the minimum value for any group.
 *   `metric.overall` reports the overall value for all groups combined, and is the same as calling `metric` directly
 *   `metric.ratio` reports the smallest values for any group divided by the largest
 *   `metric.per_group` reports the value for every group.

All of these can be passed directly to fit, or to the evaluation functions we provide.

The vast majority of fairness metrics are implemented as a `.diff` of a standard performance measure, and by placing a `.min` after any measure such as `recall` or `precision` it is possible to add constraints that enforce that the precision or recall is above a particular value for every group.
gm.

### Dataset Measures

| Name             | Definition                                                       |
|------------------|------------------------------------------------------------------|
| `gm.count`          | Total number of points in a dataset or group                     |
| `gm.pos_data_count` | Total number of positively labeled points in a dataset or group |
| `gm.neg_data_count` | Total number of negatively labeled points in a dataset or group |
| `gm.pos_data_rate`  | Ratio of positively labeled points to size of the group         |
| `gm.neg_data_rate`  | Ratio of negatively labeled points to size of the group         |

### Standard Prediction Measures

| Name             | Definition                                                                                                     |
|------------------|----------------------------------------------------------------------------------------------------------------|
| `gm.pos_pred_rate`  | Positive Prediction Rate: Ratio of the number of positively predicted points to the size of the group          |
| `gm.neg_pred_rate`  | Negative Prediction Rate: Ratio of the number of negatively predicted points to the size of the group          |
| `gm.true_pos_rate`  | True Positive Rate: Ratio of true positives divided by total positive predictions                              |
| `gm.true_neg_rate`  | True Negative Rate: Ratio of true negatives divided by total negative predictions                              |
| `gm.false_pos_rate` | False Positive Rate: Ratio of False Positives divided by total negative prediction                             |
| `gm.false_neg_rate` | False Negative Rate: Ratio of False Negatives divided by total positive predictions                            |
| `gm.pos_pred_val`   | Positive Predicted Value': Ratio of True Positives divided by the total number of points with positive label   |
| `gm.neg_pred_val`   | Negative Predicted Value': Ratio of True Negatives divided by the total number of points with a negative label |

### Core Performance Measures

| Name                | Definition                                                                                                                                                                               |
|---------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `gm.accuracy`          | Proportion of points correctly identified                                                                                                                                                 |
| `gm.balanced_accuracy` | The average of the proportion of points with a positive label correctly identified and the proportion of points with a negative label correctly identified                              |
| `gm.min_accuracy`      | The minimum of the proportion of points with a positive label correctly identified and the proportion of points with a negative label correctly identified (common in min-max fairness) |
| `gm.f1`                | F1 Score. Defined as:  (2 * TP) / (2 * TP + FP + FN)                                                                                                                                     |
| `gm.precision`         | AKA Positive Prediction Rate                                                                                                                                                             |
| `gm.recall`            | AKA True Positive Prediction Rate                                                                                                                                                        |
| `gm.mcc`               | Matthews Correlation Coefficient. See https://en.wikipedia.org/wiki/Phi_coefficient                                                                                                      |

### Additional Performance Measures

| Name              | Definition                                                                        |
|-------------------|-----------------------------------------------------------------------------------|
| `gm.acceptance_rate` | AKA precision AKA Positive Prediction Rate                                        |
| `gm.cond_accept`     | Conditional Acceptance Rate. The ratio of positive predictions to positive labels |
| `gm.cond_reject`     | Conditional Rejectance Rate. The ratio of negative predictions to negative labels |
| `gm.specificity`     | AKA True Negative Rate                                                            |
| `gm.rejection_rate`  | AKA Negative Predicted Value                                                      |
| `gm.error_ratio`     | The ratio of False Positives to False Negatives                                    |


### Fairness Measures Supported

[Sagemaker Clarify](https://docs.aws.amazon.com/sagemaker/latest/dg/clarify-measure-post-training-bias.html) Measures

| Name                   | Definition                                                                                                                                                                                                                                                                                   |
|------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `gm.class_imbalance`      | Maximum difference between groups in Positive Data Rate                                                                                                                                                                                                                                      |
| `gm.demographic_parity`   | AKA Statistical Parity.  Maximum difference between groups in Positive Prediction Rate                                                                                                                                                                                                       |
| `gm.disparate_impact`     | The smallest Positive Prediction Rate of any group divided by the largest                                                                                                                                                                                                                    |
| `gm.accuracy.diff`        | Maximum difference between groups in Accuracy                                                                                                                                                                                                                                                |
| `gm.recall.diff`          | AKA Equal Opportunity. Maximum difference between groups in Recall                                                                                                                                                                                                                           |
| `gm.cond_accept.diff`     | Maximum difference between groups in Conditional Acceptance Rate                                                                                                                                                                                                                             |
| `gm.acceptance_rate.diff` | Maximum difference between groups in Acceptance Rate                                                                                                                                                                                                                                         |
| `gm.specificity.diff`     | Maximum difference between groups in Specificity  (or True Negative Rate)                                                                                                                                                                                                                    |
| `gm.cond_reject.diff`     | Maximum difference between groups in Conditonal Rejectance Rate                                                                                                                                                                                                                              |
| `gm.rejection_rate.diff`  | Maximum difference between groups in Rejection Rate (or Negative Predicted Value)                                                                                                                                                                                                            |
| `gm.treatment_equality`   | Maximum difference between groups in Error Ratio                                                                                                                                                                                                                                             |
| `gm.gen_entropy`          | This is the expected square of a particular utility function divided by its expected value, minus 1 and then divided by 2. The function takes the form: `TP*1+FP*2+FN*1`, where TP, FP, NP, and TN are the true positives, false positives, false negatives and true negatives respectively. |

Measures from [Verma and Rubin](https://fairware.cs.umass.edu/papers/Verma.pdf).

All the measures in Verma and Rubin are defined as strict equalities for two groups. We relax them into a continuous measure that reports the maximum difference over any pair of groups between the left and right sides of the equality.
These relaxations take value 0 only if the equalities are satisfied for all pairs of groups.

| Name                     | Definition                                                                                            |
|--------------------------|-------------------------------------------------------------------------------------------------------|
| `gm.statistical_parity`  | AKA Demographic Parity. Maximum difference between groups in Positive Prediction Rate                 |
| `gm.predictive_parity`   | AKA Rejection Rate Difference. Maximum difference between groups in Precision                         |
| `gm.false_pos_rate.diff` | AKA  Specificity Difference. Maximum difference between groups in False Positive rate.                |
| `gm.false_neg_rate.diff` | AKA Equal Opportunity and Recall difference. Maximum difference between groups in False Negative Rate |
| `gm.equalized_odds`      | The average of `true_pos_rate.diff` and  `false_neg_rate.diff`                                        |
| `gm.cond_use_accuracy`   | The average of `pos_pred_val.diff` and `neg_pred_val.diff`                                            |
| `gm.predictive_equality` | Maximum difference in False Negative Rate                                                             |
| `gm.accuracy._parity`    | Maximum difference in Accuracy                                                                        |
| `gm.treatment_equality`  | Maximum difference between groups in Error Ratio                                                      |


