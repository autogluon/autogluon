# Predicting Multiple Columns in a Table (Multi-Label Prediction)
:label:`sec_tabularmultilabel`

In multi-label prediction, we wish to predict multiple columns of a table (i.e. labels) based on the values in the remaining columns. Here we present a simple strategy to do this with AutoGluon, which simply maintains a separate [TabularPredictor](../../api/autogluon.predictor.html#autogluon.tabular.TabularPredictor.fit) object for each column being predicted. Correlations between labels can be accounted for in predictions by imposing an order on the labels and allowing the `TabularPredictor` for each label to condition on the predicted values for labels that appeared earlier in the order.

### MultilabelPredictor Class

We start by defining a custom `MultilabelPredictor` class to manage a collection of `TabularPredictor` objects, one for each label. You can use the `MultilabelPredictor` similarly to an individual `TabularPredictor`, except it operates on multiple labels rather than one.

```{.python .input}
from autogluon.tabular import TabularDataset, TabularPredictor
from autogluon.common.utils.utils import setup_outputdir
from autogluon.core.utils.loaders import load_pkl
from autogluon.core.utils.savers import save_pkl
import os.path

class MultilabelPredictor():
    """ Tabular Predictor for predicting multiple columns in table.
        Creates multiple TabularPredictor objects which you can also use individually.
        You can access the TabularPredictor for a particular label via: `multilabel_predictor.get_predictor(label_i)`

        Parameters
        ----------
        labels : List[str]
            The ith element of this list is the column (i.e. `label`) predicted by the ith TabularPredictor stored in this object.
        path : str, default = None
            Path to directory where models and intermediate outputs should be saved.
            If unspecified, a time-stamped folder called "AutogluonModels/ag-[TIMESTAMP]" will be created in the working directory to store all models.
            Note: To call `fit()` twice and save all results of each fit, you must specify different `path` locations or don't specify `path` at all.
            Otherwise files from first `fit()` will be overwritten by second `fit()`.
            Caution: when predicting many labels, this directory may grow large as it needs to store many TabularPredictors.
        problem_types : List[str], default = None
            The ith element is the `problem_type` for the ith TabularPredictor stored in this object.
        eval_metrics : List[str], default = None
            The ith element is the `eval_metric` for the ith TabularPredictor stored in this object.
        consider_labels_correlation : bool, default = True
            Whether the predictions of multiple labels should account for label correlations or predict each label independently of the others.
            If True, the ordering of `labels` may affect resulting accuracy as each label is predicted conditional on the previous labels appearing earlier in this list (i.e. in an auto-regressive fashion).
            Set to False if during inference you may want to individually use just the ith TabularPredictor without predicting all the other labels.
        kwargs :
            Arguments passed into the initialization of each TabularPredictor.

    """

    multi_predictor_file = 'multilabel_predictor.pkl'

    def __init__(self, labels, path=None, problem_types=None, eval_metrics=None, consider_labels_correlation=True, **kwargs):
        if len(labels) < 2:
            raise ValueError("MultilabelPredictor is only intended for predicting MULTIPLE labels (columns), use TabularPredictor for predicting one label (column).")
        if (problem_types is not None) and (len(problem_types) != len(labels)):
            raise ValueError("If provided, `problem_types` must have same length as `labels`")
        if (eval_metrics is not None) and (len(eval_metrics) != len(labels)):
            raise ValueError("If provided, `eval_metrics` must have same length as `labels`")
        self.path = setup_outputdir(path, warn_if_exist=False)
        self.labels = labels
        self.consider_labels_correlation = consider_labels_correlation
        self.predictors = {}  # key = label, value = TabularPredictor or str path to the TabularPredictor for this label
        if eval_metrics is None:
            self.eval_metrics = {}
        else:
            self.eval_metrics = {labels[i] : eval_metrics[i] for i in range(len(labels))}
        problem_type = None
        eval_metric = None
        for i in range(len(labels)):
            label = labels[i]
            path_i = self.path + "Predictor_" + label
            if problem_types is not None:
                problem_type = problem_types[i]
            if eval_metrics is not None:
                eval_metric = eval_metrics[i]
            self.predictors[label] = TabularPredictor(label=label, problem_type=problem_type, eval_metric=eval_metric, path=path_i, **kwargs)

    def fit(self, train_data, tuning_data=None, **kwargs):
        """ Fits a separate TabularPredictor to predict each of the labels.

            Parameters
            ----------
            train_data, tuning_data : str or autogluon.tabular.TabularDataset or pd.DataFrame
                See documentation for `TabularPredictor.fit()`.
            kwargs :
                Arguments passed into the `fit()` call for each TabularPredictor.
        """
        if isinstance(train_data, str):
            train_data = TabularDataset(train_data)
        if tuning_data is not None and isinstance(tuning_data, str):
            tuning_data = TabularDataset(tuning_data)
        train_data_og = train_data.copy()
        if tuning_data is not None:
            tuning_data_og = tuning_data.copy()
        else:
            tuning_data_og = None
        save_metrics = len(self.eval_metrics) == 0
        for i in range(len(self.labels)):
            label = self.labels[i]
            predictor = self.get_predictor(label)
            if not self.consider_labels_correlation:
                labels_to_drop = [l for l in self.labels if l != label]
            else:
                labels_to_drop = [self.labels[j] for j in range(i+1, len(self.labels))]
            train_data = train_data_og.drop(labels_to_drop, axis=1)
            if tuning_data is not None:
                tuning_data = tuning_data_og.drop(labels_to_drop, axis=1)
            print(f"Fitting TabularPredictor for label: {label} ...")
            predictor.fit(train_data=train_data, tuning_data=tuning_data, **kwargs)
            self.predictors[label] = predictor.path
            if save_metrics:
                self.eval_metrics[label] = predictor.eval_metric
        self.save()

    def predict(self, data, **kwargs):
        """ Returns DataFrame with label columns containing predictions for each label.

            Parameters
            ----------
            data : str or autogluon.tabular.TabularDataset or pd.DataFrame
                Data to make predictions for. If label columns are present in this data, they will be ignored. See documentation for `TabularPredictor.predict()`.
            kwargs :
                Arguments passed into the predict() call for each TabularPredictor.
        """
        return self._predict(data, as_proba=False, **kwargs)

    def predict_proba(self, data, **kwargs):
        """ Returns dict where each key is a label and the corresponding value is the `predict_proba()` output for just that label.

            Parameters
            ----------
            data : str or autogluon.tabular.TabularDataset or pd.DataFrame
                Data to make predictions for. See documentation for `TabularPredictor.predict()` and `TabularPredictor.predict_proba()`.
            kwargs :
                Arguments passed into the `predict_proba()` call for each TabularPredictor (also passed into a `predict()` call).
        """
        return self._predict(data, as_proba=True, **kwargs)

    def evaluate(self, data, **kwargs):
        """ Returns dict where each key is a label and the corresponding value is the `evaluate()` output for just that label.

            Parameters
            ----------
            data : str or autogluon.tabular.TabularDataset or pd.DataFrame
                Data to evalate predictions of all labels for, must contain all labels as columns. See documentation for `TabularPredictor.evaluate()`.
            kwargs :
                Arguments passed into the `evaluate()` call for each TabularPredictor (also passed into the `predict()` call).
        """
        data = self._get_data(data)
        eval_dict = {}
        for label in self.labels:
            print(f"Evaluating TabularPredictor for label: {label} ...")
            predictor = self.get_predictor(label)
            eval_dict[label] = predictor.evaluate(data, **kwargs)
            if self.consider_labels_correlation:
                data[label] = predictor.predict(data, **kwargs)
        return eval_dict

    def save(self):
        """ Save MultilabelPredictor to disk. """
        for label in self.labels:
            if not isinstance(self.predictors[label], str):
                self.predictors[label] = self.predictors[label].path
        save_pkl.save(path=self.path+self.multi_predictor_file, object=self)
        print(f"MultilabelPredictor saved to disk. Load with: MultilabelPredictor.load('{self.path}')")

    @classmethod
    def load(cls, path):
        """ Load MultilabelPredictor from disk `path` previously specified when creating this MultilabelPredictor. """
        path = os.path.expanduser(path)
        if path[-1] != os.path.sep:
            path = path + os.path.sep
        return load_pkl.load(path=path+cls.multi_predictor_file)

    def get_predictor(self, label):
        """ Returns TabularPredictor which is used to predict this label. """
        predictor = self.predictors[label]
        if isinstance(predictor, str):
            return TabularPredictor.load(path=predictor)
        return predictor

    def _get_data(self, data):
        if isinstance(data, str):
            return TabularDataset(data)
        return data.copy()

    def _predict(self, data, as_proba=False, **kwargs):
        data = self._get_data(data)
        if as_proba:
            predproba_dict = {}
        for label in self.labels:
            print(f"Predicting with TabularPredictor for label: {label} ...")
            predictor = self.get_predictor(label)
            if as_proba:
                predproba_dict[label] = predictor.predict_proba(data, as_multiclass=True, **kwargs)
            data[label] = predictor.predict(data, **kwargs)
        if not as_proba:
            return data[self.labels]
        else:
            return predproba_dict
```

### Training

Let's now apply our multi-label predictor to predict multiple columns in a data table. We first train models to predict each of the labels.


```{.python .input}
train_data = TabularDataset('https://autogluon.s3.amazonaws.com/datasets/Inc/train.csv')
subsample_size = 500  # subsample subset of data for faster demo, try setting this to much larger values
train_data = train_data.sample(n=subsample_size, random_state=0)
train_data.head()
```

```{.python .input}
labels = ['education-num','education','class']  # which columns to predict based on the others
problem_types = ['regression','multiclass','binary']  # type of each prediction problem (optional)
eval_metrics = ['mean_absolute_error','accuracy','accuracy']  # metrics used to evaluate predictions for each label (optional)
save_path = 'agModels-predictEducationClass'  # specifies folder to store trained models (optional)

time_limit = 5  # how many seconds to train the TabularPredictor for each label, set much larger in your applications!
```

```{.python .input}
multi_predictor = MultilabelPredictor(labels=labels, problem_types=problem_types, eval_metrics=eval_metrics, path=save_path)
multi_predictor.fit(train_data, time_limit=time_limit)
```

### Inference and Evaluation

After training, you can easily use the `MultilabelPredictor` to predict all labels in new data:

```{.python .input}
test_data = TabularDataset('https://autogluon.s3.amazonaws.com/datasets/Inc/test.csv')
test_data = test_data.sample(n=subsample_size, random_state=0)
test_data_nolab = test_data.drop(columns=labels)  # unnecessary, just to demonstrate we're not cheating here
test_data_nolab.head()
```

```{.python .input}
multi_predictor = MultilabelPredictor.load(save_path)  # unnecessary, just demonstrates how to load previously-trained multilabel predictor from file

predictions = multi_predictor.predict(test_data_nolab)
print("Predictions:  \n", predictions)
```

We can also easily evaluate the performance of our predictions if our new data contain the ground truth labels:

```{.python .input}
evaluations = multi_predictor.evaluate(test_data)
print(evaluations)
print("Evaluated using metrics:", multi_predictor.eval_metrics)
```


### Accessing the TabularPredictor for One Label

We can also directly work with the `TabularPredictor` for any one of the labels as follows. However we recommend you set `consider_labels_correlation=False` before training if you later plan to use an individual `TabularPredictor` to predict just one label rather than all of the labels predicted by the `MultilabelPredictor`.

```{.python .input}
predictor_class = multi_predictor.get_predictor('class')
predictor_class.leaderboard(silent=True)
```

### Tips

In order to obtain the best predictions, you should generally add the following arguments to `MultilabelPredictor.fit()`:

1) Specify `eval_metrics` to the metrics you will use to evaluate predictions for each label

2) Specify `presets='best_quality'` to tell AutoGluon you care about predictive performance more than latency/memory usage, which will utilize stack ensembling when predicting each label.


If you find that too much memory/disk is being used, try calling `MultilabelPredictor.fit()` with additional arguments discussed under ["If you encounter memory issues" in the In Depth Tutorial](tabular-indepth.html#if-you-encounter-memory-issues) or ["If you encounter disk space issues"](tabular-indepth.html#if-you-encounter-disk-space-issues).

If you find inference too slow, you can try the strategies discussed under ["Accelerating Inference" in the In Depth Tutorial](tabular-indepth.html#accelerating-inference).
In particular, simply try specifying the following preset in `MultilabelPredictor.fit()`: `presets = ['good_quality', 'optimize_for_deployment']`
