# Adding a custom model to AutoGluon (Advanced)
:label:`sec_tabularcustommodeladvanced`

**Tip**: If you are new to AutoGluon, review :ref:`sec_tabularquick` to learn the basics of the AutoGluon API.

In this tutorial we will cover advanced custom model options that go beyond the topics covered in :ref:`sec_tabularcustommodel`.

It is assumed that you have fully read through :ref:`sec_tabularcustommodel` prior to this tutorial.

## Loading the data

First we will load the data. For this tutorial we will use the adult income dataset because it has a mix of integer, float, and categorical features.

```{.python .input}
from autogluon.tabular import TabularDataset

train_data = TabularDataset('https://autogluon.s3.amazonaws.com/datasets/Inc/train.csv')  # can be local CSV file as well, returns Pandas DataFrame
test_data = TabularDataset('https://autogluon.s3.amazonaws.com/datasets/Inc/test.csv')  # another Pandas DataFrame
label = 'class'  # specifies which column do we want to predict
train_data = train_data.sample(n=1000, random_state=0)  # subsample for faster demo

train_data.head(5)
```

## Force features to be passed to models without preprocessing / dropping

Reasons why you would want to do this is if you have model logic that requires a particular column to always be present,
regardless of its content. For example, if you are fine-tuning a pre-trained language model that expects
a feature indicating the language of the text in a given row which dictates how the text is preprocessed,
but training data only includes one language, without this adjustment
the language identifier feature would be dropped prior to fitting the model.

### Force features to not be dropped in model-specific preprocessing

To avoid dropping features in custom models due to having only 1 unique value,
add the following `_get_default_auxiliary_params` method to your custom model class:

```{.python .input}
from autogluon.core.models import AbstractModel

class DummyModel(AbstractModel):
    def _fit(self, X, **kwargs):
        print(f'Before {self.__class__.__name__} Preprocessing ({len(X.columns)} features):\n\t{list(X.columns)}')
        X = self.preprocess(X)
        print(f'After  {self.__class__.__name__} Preprocessing ({len(X.columns)} features):\n\t{list(X.columns)}')
        print(X.head(5))

class DummyModelKeepUnique(DummyModel):
    def _get_default_auxiliary_params(self) -> dict:
        default_auxiliary_params = super()._get_default_auxiliary_params()
        extra_auxiliary_params = dict(
            drop_unique=False,  # Whether to drop features that have only 1 unique value, default is True
        )
        default_auxiliary_params.update(extra_auxiliary_params)
        return default_auxiliary_params
```

### Force features to not be dropped in global preprocessing

While the above fix for model-specific preprocessing works if the feature is still present after global preprocessing,
it won't help if the feature was already dropped before getting to the model. For this, we need to
create a new feature generator class
which separates the preprocessing logic between normal features and user override features.

Here is an example implementation:

```{.python .input}
# WARNING: To use this in practice, you must put this code in a separate python file
#  from the main process and import it or else it will not be serializable.)
from autogluon.features import BulkFeatureGenerator, AutoMLPipelineFeatureGenerator, IdentityFeatureGenerator


class CustomFeatureGeneratorWithUserOverride(BulkFeatureGenerator):
    def __init__(self, automl_generator_kwargs: dict = None, **kwargs):
        generators = self._get_default_generators(automl_generator_kwargs=automl_generator_kwargs)
        super().__init__(generators=generators, **kwargs)

    def _get_default_generators(self, automl_generator_kwargs: dict = None):
        if automl_generator_kwargs is None:
            automl_generator_kwargs = dict()

        generators = [
            [
                # Preprocessing logic that handles normal features
                AutoMLPipelineFeatureGenerator(banned_feature_special_types=['user_override'], **automl_generator_kwargs),

                # Preprocessing logic that handles special features user wishes to treat separately, here we simply skip preprocessing for these features.
                IdentityFeatureGenerator(infer_features_in_args=dict(required_special_types=['user_override'])),
            ],
        ]
        return generators
```

The above code splits the preprocessing logic of a feature
depending on if it is tagged with the `'user_override'` special type in feature metadata.
To tag three features `['age', 'native-country', 'dummy_feature']` in this way,
you can do the following:

```{.python .input}
# add a useless dummy feature to show that it is not dropped in preprocessing
train_data['dummy_feature'] = 'dummy value'
test_data['dummy_feature'] = 'dummy value'

from autogluon.tabular import FeatureMetadata
feature_metadata = FeatureMetadata.from_df(train_data)

print('Before inserting overrides:')
print(feature_metadata)

feature_metadata = feature_metadata.add_special_types(
    {
        'age': ['user_override'],
        'native-country': ['user_override'],
        'dummy_feature': ['user_override'],
    }
)

print('After inserting overrides:')
print(feature_metadata)
```

Note that this is only one example implementation of a custom feature generator that has bifurcated preprocessing logic.
Users can make their tagging and feature generator logic arbitrarily complex to fit their needs.
In this example, we perform the standard preprocessing on non-tagged features, and for tagged features we pass
them through `IdentityFeatureGenerator` which is a no-op logic that does not alter the features in any way.
Instead of an `IdentityFeatureGenerator`, you could use any kind of feature generator to suite your needs.

### Putting it all together

```{.python .input}
# Separate features and labels
X = train_data.drop(columns=[label])
y = train_data[label]
X_test = test_data.drop(columns=[label])
y_test = test_data[label]

# preprocess the label column, as done in the prior custom model tutorial
from autogluon.core.data import LabelCleaner
from autogluon.core.utils import infer_problem_type
# Construct a LabelCleaner to neatly convert labels to float/integers during model training/inference, can also use to inverse_transform back to original.
problem_type = infer_problem_type(y=y)  # Infer problem type (or else specify directly)
label_cleaner = LabelCleaner.construct(problem_type=problem_type, y=y)
y_preprocessed = label_cleaner.transform(y)
y_test_preprocessed = label_cleaner.transform(y_test)

# Make sure to specify your custom feature metadata to the feature generator
my_custom_feature_generator = CustomFeatureGeneratorWithUserOverride(feature_metadata_in=feature_metadata)

X_preprocessed = my_custom_feature_generator.fit_transform(X)
X_test_preprocessed = my_custom_feature_generator.transform(X_test)
```

Notice how the user_override features were not preprocessed:

```{.python .input}
print(list(X_preprocessed.columns))
X_preprocessed.head(5)
```

Now lets see what happens when we send this data to fit a dummy model:

```{.python .input}
dummy_model = DummyModel()
dummy_model.fit(X=X, y=y, feature_metadata=my_custom_feature_generator.feature_metadata)
```

Notice how the model dropped `dummy_feature` during the preprocess call. Now lets see what happens if we use `DummyModelKeepUnique`:

```{.python .input}
dummy_model_keep_unique = DummyModelKeepUnique()
dummy_model_keep_unique.fit(X=X, y=y, feature_metadata=my_custom_feature_generator.feature_metadata)
```

Now `dummy_feature` is no longer dropped!

The above code logic can be re-used for testing your own complex model implementations,
simply replace `DummyModelKeepUnique` with your custom model and check that it keeps the features you want to use.

### Keeping Features via TabularPredictor

Now let's demonstrate how to do this via TabularPredictor in far fewer lines of code.
Note that this code will raise an exception if ran in this tutorial because the
custom model and feature generator must exist in other files for them to be serializable.
Therefore, we will not run the code in the tutorial.
(It will also raise an exception because DummyModel isn't a real model)

```
from autogluon.tabular import TabularPredictor

feature_generator = CustomFeatureGeneratorWithUserOverride()
predictor = TabularPredictor(label=label)
predictor.fit(
    train_data=train_data,
    feature_metadata=feature_metadata,  # feature metadata with your overrides
    feature_generator=feature_generator,  # your custom feature generator that handles the overrides
    hyperparameters={
        'GBM': {},  # Can fit your custom model alongside default models
        DummyModel: {},  # Will drop dummy_feature
        DummyModelKeepUnique: {},  # Will not drop dummy_feature
        # DummyModel: {'ag_args_fit': {'drop_unique': False}},  # This is another way to get same result as using DummyModelKeepUnique
    }
)
```
