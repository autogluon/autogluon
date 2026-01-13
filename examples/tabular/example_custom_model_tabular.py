"""Example script for defining and using custom models in AutoGluon Tabular"""

from autogluon.core.data import LabelCleaner
from autogluon.core.models import AbstractModel
from autogluon.core.utils import infer_problem_type
from autogluon.tabular import TabularDataset, TabularPredictor
from autogluon.tabular.configs.hyperparameter_configs import get_hyperparameter_config

#########################
# Create a custom model #
#########################


# In this example, we create a custom Naive Bayes model for use in AutoGluon
class NaiveBayesModel(AbstractModel):
    # The `_preprocess` method takes the input data and transforms it to the internal representation usable by the model.
    # `_preprocess` is called by `preprocess` and is used during model fit and model inference.
    def _preprocess(self, X, **kwargs):
        # Drop category and object column dtypes, since NaiveBayes can't handle these dtypes.
        cat_columns = X.select_dtypes(["category", "object"]).columns
        X = X.drop(cat_columns, axis=1)
        # Add a fillna call to handle missing values.
        return super()._preprocess(X, **kwargs).fillna(0)

    # The `_fit` method takes the input training data (and optionally the validation data) and trains the model.
    def _fit(self, X, y, **kwargs):
        from sklearn.naive_bayes import GaussianNB

        # It is important to call `preprocess(X)` in `_fit` to replicate what will occur during inference.
        X = self.preprocess(X)
        self.model = GaussianNB(**self.params)
        self.model.fit(X, y)


# Example of a more optimized implementation that drops the invalid features earlier on to avoid having to make repeated checks.
class AdvancedNaiveBayesModel(AbstractModel):
    def _preprocess(self, X, **kwargs):
        # Add a fillna call to handle missing values.
        return super()._preprocess(X, **kwargs).fillna(0)

    def _fit(self, X, y, **kwargs):
        from sklearn.naive_bayes import GaussianNB

        X = self.preprocess(X)
        self.model = GaussianNB(**self.params)
        self.model.fit(X, y)

    # The `_get_default_auxiliary_params` method defines various model-agnostic parameters such as maximum memory usage and valid input column dtypes.
    # For most users who build custom models, they will only need to specify the valid/invalid dtypes to the model here.
    def _get_default_auxiliary_params(self) -> dict:
        default_auxiliary_params = super()._get_default_auxiliary_params()
        extra_auxiliary_params = dict(
            # Drop category and object column dtypes, since NaiveBayes can't handle these dtypes.
            ignored_type_group_raw=["category", "object"],
        )
        default_auxiliary_params.update(extra_auxiliary_params)
        return default_auxiliary_params


################
# Loading Data #
################

train_data = TabularDataset(
    "https://autogluon.s3.amazonaws.com/datasets/Inc/train.csv"
)  # can be local CSV file as well, returns Pandas DataFrame
test_data = TabularDataset("https://autogluon.s3.amazonaws.com/datasets/Inc/test.csv")  # another Pandas DataFrame
label = "class"  # specifies which column do we want to predict
train_data = train_data.head(1000)  # subsample for faster demo

#####################################################
# Training custom model outside of TabularPredictor #
#####################################################

# Separate features and labels
X = train_data.drop(columns=[label])
y = train_data[label]

# Construct a LabelCleaner to neatly convert labels to float/integers during model training/inference, can also use to inverse_transform back to original.
problem_type = infer_problem_type(y=y)  # Infer problem type (or else specify directly)
label_cleaner = LabelCleaner.construct(problem_type=problem_type, y=y)
y_clean = label_cleaner.transform(y)

naive_bayes_model = NaiveBayesModel()
naive_bayes_model.fit(X=X, y=y_clean)  # Fit custom model

# To save to disk and load the model, do the following:
# load_path = naive_bayes_model.path
# naive_bayes_model.save()
# del naive_bayes_model
# naive_bayes_model = NaiveBayesModel.load(path=load_path)

# Prepare test data
X_test = test_data.drop(columns=[label])
y_test = test_data[label]
y_test_clean = label_cleaner.transform(y_test)

y_pred = naive_bayes_model.predict(X_test)
print(y_pred)
y_pred_orig = label_cleaner.inverse_transform(y_pred)
print(y_pred_orig)

score = naive_bayes_model.score(X_test, y_test_clean)
print(f"test score ({naive_bayes_model.eval_metric.name}) = {score}")

################################################
# Training custom model using TabularPredictor #
################################################

custom_hyperparameters = {NaiveBayesModel: {}}
# custom_hyperparameters = {NaiveBayesModel: [{}, {'var_smoothing': 0.00001}, {'var_smoothing': 0.000002}]}  # Train 3 NaiveBayes models with different hyperparameters
predictor = TabularPredictor(label=label).fit(
    train_data, hyperparameters=custom_hyperparameters
)  # Train a single default NaiveBayesModel
predictor.leaderboard(test_data)

y_pred = predictor.predict(test_data)
print(y_pred)

#######################################################################
# Training custom model alongside other models using TabularPredictor #
#######################################################################

# Now we add the custom model to be trained alongside the default models:
custom_hyperparameters.update(get_hyperparameter_config("default"))
predictor = TabularPredictor(label=label).fit(
    train_data, hyperparameters=custom_hyperparameters
)  # Train the default models plus a single default NaiveBayesModel
# predictor = TabularPredictor(label=label).fit(train_data, hyperparameters=custom_hyperparameters, auto_stack=True)  # We can even use the custom model in a multi-layer stack ensemble
predictor.leaderboard(test_data)

y_pred = predictor.predict(test_data)
print(y_pred)
