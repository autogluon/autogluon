# Adding a custom model to AutoGluon
:label:`sec_tabularcustommodel`

**Tip**: If you are new to AutoGluon, review :ref:`sec_tabularquick` to learn the basics of the AutoGluon API.

This tutorial describes how to add a custom model to AutoGluon that can be trained, hyperparameter-tuned, and ensembled alongside the default models ([default model documentation](../../api/autogluon.tabular.models.html#module-autogluon.tabular.models)).

In this example, we create a custom Random Forest model for use in AutoGluon. All models in AutoGluon inherit from the AbstractModel class ([AbstractModel source code](../../_modules/autogluon/core/models/abstract/abstract_model.html)), and must follow its API to work alongside other models.

Note that while this tutorial provides a basic model implementation, this does not cover many aspects that are used in most implemented models.

To best understand how to implement more advanced functionality, refer to the [source code](../../api/autogluon.tabular.models.html#module-autogluon.tabular.models) of the following models:

| Functionality | Reference Implementation |
| ------------- | ------------------------ |
| Respecting time limit / early stopping logic | [LGBModel](../../_modules/autogluon/tabular/models/lgb/lgb_model.html#LGBModel) and [RFModel](../../_modules/autogluon/tabular/models/rf/rf_model.html#RFModel)
| Respecting memory usage limit | LGBModel and RFModel
| Sample weight support | LGBModel
| Validation data and eval_metric usage | LGBModel
| GPU training support | LGBModel
| Save / load logic of non-serializable models | [NNFastAiTabularModel](../../_modules/autogluon/tabular/models/fastainn/tabular_nn_fastai.html#NNFastAiTabularModel)
| Advanced problem type support (Softclass, Quantile) | RFModel
| Text feature type support | [TextPredictorModel](../../_modules/autogluon/tabular/models/text_prediction/text_prediction_v1_model.html#TextPredictorModel)
| Image feature type support | [ImagePredictorModel](../../_modules/autogluon/tabular/models/image_prediction/image_predictor.html#ImagePredictorModel)
| Lazy import of package dependencies | LGBModel
| Custom HPO logic | LGBModel

## Implementing a custom model

Here we define the custom model we will be working with for the rest of the tutorial.

The most important methods that must be implemented are `_fit` and `_preprocess`.

To compare with the official AutoGluon Random Forest implementation, see the [RFModel](../../_modules/autogluon/tabular/models/rf/rf_model.html#RFModel) source code.

Follow along with the code comments to better understand how the code works.

```{.python .input}
import numpy as np
import pandas as pd

from autogluon.core.models import AbstractModel
from autogluon.features.generators import LabelEncoderFeatureGenerator

class CustomRandomForestModel(AbstractModel):
    def __init__(self, **kwargs):
        # Simply pass along kwargs to parent, and init our internal `_feature_generator` variable to None
        super().__init__(**kwargs)
        self._feature_generator = None

    # The `_preprocess` method takes the input data and transforms it to the internal representation usable by the model.
    # `_preprocess` is called by `preprocess` and is used during model fit and model inference.
    def _preprocess(self, X: pd.DataFrame, is_train=False, **kwargs) -> np.ndarray:
        print(f'Entering the `_preprocess` method: {len(X)} rows of data (is_train={is_train})')
        X = super()._preprocess(X, **kwargs)

        if is_train:
            # X will be the training data.
            self._feature_generator = LabelEncoderFeatureGenerator(verbosity=0)
            self._feature_generator.fit(X=X)
        if self._feature_generator.features_in:
            # This converts categorical features to numeric via stateful label encoding.
            X = X.copy()
            X[self._feature_generator.features_in] = self._feature_generator.transform(X=X)
        # Add a fillna call to handle missing values.
        # Some algorithms will be able to handle NaN values internally (LightGBM).
        # In those cases, you can simply pass the NaN values into the inner model.
        # Finally, convert to numpy for optimized memory usage and because sklearn RF works with raw numpy input.
        return X.fillna(0).to_numpy(dtype=np.float32)

    # The `_fit` method takes the input training data (and optionally the validation data) and trains the model.
    def _fit(self,
             X: pd.DataFrame,  # training data
             y: pd.Series,  # training labels
             # X_val=None,  # val data (unused in RF model)
             # y_val=None,  # val labels (unused in RF model)
             # time_limit=None,  # time limit in seconds (ignored in tutorial)
             **kwargs):  # kwargs includes many other potential inputs, refer to AbstractModel documentation for details
        print('Entering the `_fit` method')

        # First we import the required dependencies for the model. Note that we do not import them outside of the method.
        # This enables AutoGluon to be highly extensible and modular.
        # For an example of best practices when importing model dependencies, refer to LGBModel.
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

        # Valid self.problem_type values include ['binary', 'multiclass', 'regression', 'quantile', 'softclass']
        if self.problem_type in ['regression', 'softclass']:
            model_cls = RandomForestRegressor
        else:
            model_cls = RandomForestClassifier

        # Make sure to call preprocess on X near the start of `_fit`.
        # This is necessary because the data is converted via preprocess during predict, and needs to be in the same format as during fit.
        X = self.preprocess(X, is_train=True)
        # This fetches the user-specified (and default) hyperparameters for the model.
        params = self._get_model_params()
        print(f'Hyperparameters: {params}')
        # self.model should be set to the trained inner model, so that internally during predict we can call `self.model.predict(...)`
        self.model = model_cls(**params)
        self.model.fit(X, y)
        print('Exiting the `_fit` method')

    # The `_set_default_params` method defines the default hyperparameters of the model.
    # User-specified parameters will override these values on a key-by-key basis.
    def _set_default_params(self):
        default_params = {
            'n_estimators': 300,
            'n_jobs': -1,
            'random_state': 0,
        }
        for param, val in default_params.items():
            self._set_default_param_value(param, val)

    # The `_get_default_auxiliary_params` method defines various model-agnostic parameters such as maximum memory usage and valid input column dtypes.
    # For most users who build custom models, they will only need to specify the valid/invalid dtypes to the model here.
    def _get_default_auxiliary_params(self) -> dict:
        default_auxiliary_params = super()._get_default_auxiliary_params()
        extra_auxiliary_params = dict(
            # the total set of raw dtypes are: ['int', 'float', 'category', 'object', 'datetime']
            # object feature dtypes include raw text and image paths, which should only be handled by specialized models
            # datetime raw dtypes are generally converted to int in upstream pre-processing,
            # so models generally shouldn't need to explicitly support datetime dtypes.
            valid_raw_types=['int', 'float', 'category'],
            # Other options include `valid_special_types`, `ignored_type_group_raw`, and `ignored_type_group_special`.
            # Refer to AbstractModel for more details on available options.
        )
        default_auxiliary_params.update(extra_auxiliary_params)
        return default_auxiliary_params

```

## Loading the data

Next we will load the data. For this tutorial we will use the adult income dataset because it has a mix of integer, float, and categorical features.

```{.python .input}
from autogluon.tabular import TabularDataset

train_data = TabularDataset('https://autogluon.s3.amazonaws.com/datasets/Inc/train.csv')  # can be local CSV file as well, returns Pandas DataFrame
test_data = TabularDataset('https://autogluon.s3.amazonaws.com/datasets/Inc/test.csv')  # another Pandas DataFrame
label = 'class'  # specifies which column do we want to predict
train_data = train_data.sample(n=1000, random_state=0)  # subsample for faster demo

train_data.head(5)
```

## Training a custom model without TabularPredictor

Below we will demonstrate how to train the model outside [TabularPredictor](../../api/autogluon.predictor.html#module-0). This is useful for debugging and minimizing the amount of code you need to understand while implementing the model.

This process is similar to what happens internally when calling fit on `TabularPredictor`, but is simplified and minimal.

If the data was already cleaned (all numeric), then we could call fit directly with the data, but the adult dataset is not.

### Clean labels

The first step to making the input data as valid input to the model is to clean the labels.

Currently, they are strings, but we need to convert them to numeric values (0 and 1) for binary classification.

Luckily, AutoGluon already implements logic to both detect that this is binary classification (via `infer_problem_type`), and a converter to map the labels to 0 and 1 (`LabelCleaner`):

```{.python .input}
# Separate features and labels
X = train_data.drop(columns=[label])
y = train_data[label]
X_test = test_data.drop(columns=[label])
y_test = test_data[label]

from autogluon.core.data import LabelCleaner
from autogluon.core.utils import infer_problem_type
# Construct a LabelCleaner to neatly convert labels to float/integers during model training/inference, can also use to inverse_transform back to original.
problem_type = infer_problem_type(y=y)  # Infer problem type (or else specify directly)
label_cleaner = LabelCleaner.construct(problem_type=problem_type, y=y)
y_clean = label_cleaner.transform(y)

print(f'Labels cleaned: {label_cleaner.inv_map}')
print(f'inferred problem type as: {problem_type}')
print('Cleaned label values:')
y_clean.head(5)
```

### Clean features

Next, we need to clean the features. Currently, features like 'workclass' are object dtypes (strings), but we actually want to use them as categorical features. Most models won't accept string inputs, so we need to convert the strings to numbers.

AutoGluon contains an entire module dedicated to cleaning, transforming, and generating features called [autogluon.features](../../api/autogluon.features.html). Here we will use the same feature generator used internally by `TabularPredictor` to convert the object dtypes to categorical and minimize memory usage.

```{.python .input}
from autogluon.common.utils.log_utils import set_logger_verbosity
from autogluon.features.generators import AutoMLPipelineFeatureGenerator
set_logger_verbosity(2)  # Set logger so more detailed logging is shown for tutorial

feature_generator = AutoMLPipelineFeatureGenerator()
X_clean = feature_generator.fit_transform(X)

X_clean.head(5)
```

[AutoMLPipelineFeatureGenerator](../../api/autogluon.features.html#automlpipelinefeaturegenerator) does not fill missing values for numeric features nor does it rescale the values of numeric features or one-hot encode categoricals. If a model requires these operations, you'll need to add these operations into your `_preprocess` method, and may find some FeatureGenerator classes useful for this.

### Fit model

We are now ready to fit the model with the cleaned features and labels.

```{.python .input}
custom_model = CustomRandomForestModel()
# We could also specify hyperparameters to override defaults
# custom_model = CustomRandomForestModel(hyperparameters={'max_depth': 10})
custom_model.fit(X=X_clean, y=y_clean)  # Fit custom model

# To save to disk and load the model, do the following:
# load_path = custom_model.path
# custom_model.save()
# del custom_model
# custom_model = CustomRandomForestModel.load(path=load_path)
```

### Predict with trained model

Now that the model is fit, we can make predictions on new data. Remember that we need to perform the same data and label transformations to the new data as we did to the training data.

```{.python .input}
# Prepare test data
X_test_clean = feature_generator.transform(X_test)
y_test_clean = label_cleaner.transform(y_test)

X_test.head(5)
```

Get raw predictions from the test data

```{.python .input}
y_pred = custom_model.predict(X_test_clean)
print(y_pred[:5])
```

Note that these predictions are of the positive class (whichever class was inferred to 1). To get more interpretable results, do the following:

```{.python .input}
y_pred_orig = label_cleaner.inverse_transform(y_pred)
y_pred_orig.head(5)
```

### Score with trained model

By default, the model has an eval_metric specific to the problem_type. For binary classification, it uses accuracy.

We can get the accuracy score of the model by doing the following:

```{.python .input}
score = custom_model.score(X_test_clean, y_test_clean)
print(f'Test score ({custom_model.eval_metric.name}) = {score}')
```

## Training a bagged custom model without TabularPredictor

Some of the more advanced functionality in AutoGluon such as bagging can be done very easily to models once they inherit from AbstractModel.

You can even bag your custom model in a couple lines of code. This is a quick way to get quality improvements on nearly any model:

```{.python .input}
from autogluon.core.models import BaggedEnsembleModel
bagged_custom_model = BaggedEnsembleModel(CustomRandomForestModel())
# Parallel folding currently doesn't work with a class not defined in a separate module because of underlying pickle serialization issue
# You don't need this following line if you put your custom model in a separate file and import it.
bagged_custom_model.params['fold_fitting_strategy'] = 'sequential_local' 
bagged_custom_model.fit(X=X_clean, y=y_clean, k_fold=10)  # Perform 10-fold bagging
bagged_score = bagged_custom_model.score(X_test_clean, y_test_clean)
print(f'Test score ({bagged_custom_model.eval_metric.name}) = {bagged_score} (bagged)')
print(f'Bagging increased model accuracy by {round(bagged_score - score, 4) * 100}%!')
```

Note that the bagged model trained 10 CustomRandomForestModels on different splits of the training data. When making a prediction, the bagged model averages the predictions from these 10 models.

## Training a custom model with TabularPredictor

While not using [TabularPredictor](../../api/autogluon.predictor.html#module-0) allows us to simplify the amount of code we need to worry about while developing and debugging our model, eventually we want to leverage TabularPredictor to get the most out of our model.

The code to train the model from the raw data is very simple when using TabularPredictor. There is no need to specify a LabelCleaner, FeatureGenerator, or a validation set, all of that is handled internally.

Here we train 3 CustomRandomForestModel with different hyperparameters.

```{.python .input}
from autogluon.tabular import TabularPredictor

# custom_hyperparameters = {CustomRandomForestModel: {}}  # train 1 CustomRandomForestModel Model with default hyperparameters
custom_hyperparameters = {CustomRandomForestModel: [{}, {'max_depth': 10}, {'max_features': 0.9, 'max_depth': 20}]}  # Train 3 CustomRandomForestModel with different hyperparameters
predictor = TabularPredictor(label=label).fit(train_data, hyperparameters=custom_hyperparameters)
```

### Predictor leaderboard

Here we show the stats of each of the models trained. Notice that a WeightedEnsemble model was also trained. This model tries to combine the predictions of the other models to get a better validation score via ensembling.

```{.python .input}
predictor.leaderboard(test_data, silent=True)
```

### Predict with fit predictor

Here we predict with the fit predictor. This will automatically use the best model (the one with highest score_val) to predict.

```{.python .input}
y_pred = predictor.predict(test_data)
# y_pred = predictor.predict(test_data, model='CustomRandomForestModel_3')  # If we want a specific model to predict
y_pred.head(5)
```

## Hyperparameter tuning a custom model with TabularPredictor

We can easily hyperparameter tune custom models by specifying a hyperparameter search space in-place of exact values.

Here we hyperparameter tune the custom model for 20 seconds:

```{.python .input}
from autogluon.core.space import Categorical, Int, Real
custom_hyperparameters_hpo = {CustomRandomForestModel: {
    'max_depth': Int(lower=5, upper=30),
    'max_features': Real(lower=0.1, upper=1.0),
    'criterion': Categorical('gini', 'entropy'),
}}
# Hyperparameter tune CustomRandomForestModel for 20 seconds
predictor = TabularPredictor(label=label).fit(train_data,
                                              hyperparameters=custom_hyperparameters_hpo,
                                              hyperparameter_tune_kwargs='auto',  # enables HPO
                                              time_limit=20)
```

### Predictor leaderboard (HPO)

The leaderboard for the HPO run will show models with suffix `'/Tx'` in their name. This indicates the HPO trial they were performed in.

```{.python .input}
leaderboard_hpo = predictor.leaderboard(silent=True)
leaderboard_hpo
```

### Getting the hyperparameters of a trained model

Let's get the hyperparameters of the model with the highest validation score.

```{.python .input}
best_model_name = leaderboard_hpo[leaderboard_hpo['stack_level'] == 1]['model'].iloc[0]

predictor_info = predictor.info()
best_model_info = predictor_info['model_info'][best_model_name]

print(best_model_info)

print(f'Best Model Hyperparameters ({best_model_name}):')
print(best_model_info['hyperparameters'])
```

## Training a custom model alongside other models with TabularPredictor

Finally, we will train the custom model (with tuned hyperparameters) alongside the default AutoGluon models.

All this requires is getting the hyperparameter dictionary of the default models via `get_hyperparameter_config`, and adding CustomRandomForestModel as a key.

```{.python .input}
from autogluon.tabular.configs.hyperparameter_configs import get_hyperparameter_config

# Now we can add the custom model with tuned hyperparameters to be trained alongside the default models:
custom_hyperparameters = get_hyperparameter_config('default')

custom_hyperparameters[CustomRandomForestModel] = best_model_info['hyperparameters']

print(custom_hyperparameters)
```

```{.python .input}
predictor = TabularPredictor(label=label).fit(train_data, hyperparameters=custom_hyperparameters)  # Train the default models plus a single tuned CustomRandomForestModel
# predictor = TabularPredictor(label=label).fit(train_data, hyperparameters=custom_hyperparameters, presets='best_quality')  # We can even use the custom model in a multi-layer stack ensemble
predictor.leaderboard(test_data, silent=True)
```

## Wrapping up

That's all it takes to add a custom model to AutoGluon. If you create a custom model, consider [submitting a PR](https://github.com/awslabs/autogluon/pulls) so that we can add it officially to AutoGluon!

For more tutorials, refer to :ref:`sec_tabularquick` and :ref:`sec_tabularadvanced`.
