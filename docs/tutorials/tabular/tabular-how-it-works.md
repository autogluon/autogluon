# How It Works

AutoML is usually associated with Hyperparameter Optimization (HPO) of one or multiple models. Or with the automatic selection of models based on the given problem. In fact, most AutoML frameworks do this.

AutoGluon is different because it doesn't rely on HPO to achieve great performance[^1]. It's based on three main principles: (1) training a variety of **different models**, (2) using **bagging** when training those models, and (3) **stack-ensembling** those models to combine their predictive power into a "super" model. The following sections describe these concepts in more detail.

## Bagging

[Bagging (Bootstrap Aggregation)](https://www.ibm.com/topics/bagging) is a technique used in Machine Learning to improve the stability and accuracy of algorithms. The key idea is that combining multiple models usually leads to better performance than any single model because it reduces [overfitting](https://aws.amazon.com/what-is/overfitting/) and adds robustness to the prediction.

In general, the _bootstrap_ portion of bagging involves taking many random sub-samples (with replacement, i.e. the same data point can appear more than once in a sample) from the training dataset. And then training different models on the _bootstrapped_ samples.

However, AutoGluon performs bagging in a different way by combining it with **cross-validation**. In addition to the benefits of bagging, cross-validation allows us to train and validate multiple models using _all_ the training data. This also increases our confidence in the scores of the trained models. Here's how the technique works:

```{image} https://raw.githubusercontent.com/Innixma/autogluon-doc-utils/main/docs/tutorials/tabular/how-it-works/autogluon-bagging.png
:width: 900
```

1. **Partitioning:** The training data is partitioned into _K_ folds (or subsets of the dataset)[^2].
2. **Model Training:** For each of the folds, a model is trained using all the data _except_ the fold. This means we train _K_ separate model instances with different portions of the data. This is known as a _bagged_ model.
3. **Cross-validation:** Each model instance is evaluated against the hold-out fold that wasn't used during training. We then concatenate the predictions[^3] from the folds to create the **out-of-fold (OOF) predictions**. We calculate the final model cross-validation score by computing the evaluation metric using the OOF predictions and the target ground truth. Make sure to form a solid understanding of what out-of-fold (OOF) predictions are, as they are the most critical component to making stack ensembling work (_see below_).
4. **Aggregation:** At prediction time, bagging takes all these individual models and averages their predictions to generate a final answer (e.g. the class in the case of classification problems).

This same process is repeated for each of the models that AutoGluon uses. Thus, the number of models that AutoGluon trains during this process is _N x K_, where _N_ is the number of models and _K_ is the number of folds.

## Stacked Ensembling

In the most general sense, ensembling is another technique used in Machine Learning to improve the accuracy of predictions by combining the strengths of multiple models. There are multiple ways to perform ensembling[^4] and this [guide](https://web.archive.org/web/20210727094233/http://mlwave.com/kaggle-ensembling-guide/) is a great introduction to many of them.

AutoGluon, in particular, uses stack ensembling. At a high level, stack ensembling is a technique that leverages the predictions of models as extra features in the data. Here's how the technique works:

```{image} https://raw.githubusercontent.com/Innixma/autogluon-doc-utils/main/docs/tutorials/tabular/how-it-works/autogluon-stacked-ensembling.png
:width: 900
```

1. **Layer(s) of Models:** Stacked ensembling is like a multi-layer cake. Each layer consists of several different bagged models (_see above_) that use the predictions from the previous layer as inputs (features) **in addition to the original features from the training data** (akin to a skip connection). The first layer (also known as the _base_) uses only the original features from the training data.
2. **Weighted Ensemble:** The last layer consists of a single "super" model that combines the predictions from the second to last layer[^5]. The job of this model, commonly known as the _meta-model_, is to learn how to _combine_ the outputs from all previous models to make a final prediction. Think of this model as a _leader_ who makes a final decision by _weighting_ everyone else's inputs. In fact, that is exactly what AutoGluon does: it uses a [Greedy Weighted Ensemble](https://www.cs.cornell.edu/~alexn/papers/shotgun.icml04.revised.rev2.pdf) algorithm to produce the final meta-model.
3. **Residual Connections:** Note that the structure of stacked ensembling resembles that of a **Neural Network**. Therefore, advanced techniques (e.g. dropout, skip connections, etc.) used for Neural Networks could also be applied here as well.
4. **How to Train:** During training time it is critical to avoid [data leakage](https://www.kaggle.com/code/alexisbcook/data-leakage), and therefore we use the **out-of-fold (OOF) predictions** of each bag model instead of predicting directly on the train data with the bagged model. By using out-of-fold predictions, we ensure that each instance in the training dataset has a corresponding prediction that was generated by a model that **did not train** on that instance. This setup mirrors how the final ensemble model will operate on new, unseen data.
5. **How to Infer:** During inference time, we don't need to worry about data leakage, and we simply **average the predictions** of all models in a bag to generate its predictions.

Considering both bagging and stacked ensembling, the final number of models that AutoGluon trains is _M x N x K + 1_, where:

- _M_ is the number of layers in the ensemble, including the base and excluding the last layer
- _N_ is the number of models per layer
- _K_ is the number of folds for bagging
- _1_ is the final meta-model (weighted ensemble)

## What Models To Use

One key part of this whole process is deciding which models to train and ensemble. Although ensembling, in general, is a very powerful technique; choosing the right models can be the difference between mediocre and excellent performance.

To answer this question, we evaluated **1,310 models** on **200 distinct datasets** to compare the performance of different combinations of algorithms and hyperparameter configurations. The evaluation is available in this [repository](https://github.com/autogluon/tabrepo/tree/main).

With the results of this extensive evaluation, we chose a set of pre-defined configurations to use in AutoGluon by default[^6] based on the desired performance (e.g. "best quality", "medium quality", etc.). These presets even define the order in which models should be trained to maximize the use of training time.

[^1]: Hyperparameter Optimization is also possible in AutoGluon, but it's turned off by default. See the official [docs](https://auto.gluon.ai/stable/tutorials/tabular/tabular-indepth.html#specifying-hyperparameters-and-tuning-them) for more information.

[^2]: For the best quality and performance, AutoGluon uses 8 folds.

[^3]: In most cases, when we say "predictions", we are referring to "prediction probabilities" for classification and "predictions" for regression.

[^4]: In fact, Bagging _is_ a form of ensembling.

[^5]: AutoGluon goes a step further and uses a skip connection for the final layer's weighted ensemble to connect it to all previous layers at the same time to improve the results.

[^6]: Users can still pass custom hyperparameters and custom models. See the official [docs](https://auto.gluon.ai/stable/tutorials/tabular/advanced/tabular-custom-model.html) for more information.
