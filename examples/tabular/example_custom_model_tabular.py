""" Example script for defining and using custom models in AutoGluon Tabular """
import time

from autogluon import TabularPrediction as task
from autogluon.task.tabular_prediction.hyperparameter_configs import get_hyperparameter_config
from autogluon.utils.tabular.data.label_cleaner import LabelCleaner
from autogluon.utils.tabular.ml.models.abstract.abstract_model import AbstractModel
from autogluon.utils.tabular.ml.utils import infer_problem_type

#########################
# Create a custom model #
#########################

# In this example, we create a custom Naive Bayes model for use in AutoGluon
class NaiveBayesModel(AbstractModel):
    def preprocess(self, X):
        cat_columns = X.select_dtypes(['category', 'object']).columns
        X = X.drop(cat_columns, axis=1)
        return super().preprocess(X).fillna(0)

    def _fit(self, X_train, y_train, **kwargs):
        from sklearn.naive_bayes import GaussianNB
        X_train = self.preprocess(X_train)
        self.model = GaussianNB(**self.params)
        self.model.fit(X_train, y_train)


################
# Loading Data #
################

train_data = task.Dataset(file_path='https://autogluon.s3.amazonaws.com/datasets/Inc/train.csv')  # can be local CSV file as well, returns Pandas DataFrame
test_data = task.Dataset(file_path='https://autogluon.s3.amazonaws.com/datasets/Inc/test.csv')  # another Pandas DataFrame
label_column = 'class'  # specifies which column do we want to predict
train_data = train_data.head(1000)  # subsample for faster demo

#############################################
# Training custom model outside of task.fit #
#############################################

# Separate features and labels
X_train = train_data.drop(columns=[label_column])
y_train = train_data[label_column]

problem_type = infer_problem_type(y=y_train)  # Infer problem type (or else specify directly)
naive_bayes_model = NaiveBayesModel(path='AutogluonModels/', name='CustomNaiveBayes', problem_type=problem_type)

# Construct a LabelCleaner to neatly convert labels to float/integers during model training/inference, can also use to inverse_transform back to original.
label_cleaner = LabelCleaner.construct(problem_type=problem_type, y=y_train)
y_train_clean = label_cleaner.transform(y_train)

naive_bayes_model.fit(X_train=X_train, y_train=y_train_clean)  # Fit custom model

# To save to disk and load the model, do the following:
# load_path = naive_bayes_model.path
# naive_bayes_model.save()
# del naive_bayes_model
# naive_bayes_model = NaiveBayesModel.load(path=load_path)

# Prepare test data
X_test = test_data.drop(columns=[label_column])
y_test = test_data[label_column]
y_test_clean = label_cleaner.transform(y_test)

y_pred = naive_bayes_model.predict(X_test)
print(y_pred)
y_pred_orig = label_cleaner.inverse_transform(y_pred)
print(y_pred_orig)

score = naive_bayes_model.score(X_test, y_test_clean)
print(f'test score ({naive_bayes_model.eval_metric.name}) = {score}')

########################################
# Training custom model using task.fit #
########################################

custom_hyperparameters = {NaiveBayesModel: {}}
# custom_hyperparameters = {NaiveBayesModel: [{}, {'var_smoothing': 0.00001}, {'var_smoothing': 0.000002}]}  # Train 3 NaiveBayes models with different hyperparameters
predictor = task.fit(train_data=train_data, label=label_column, hyperparameters=custom_hyperparameters)  # Train a single default NaiveBayesModel
predictor.leaderboard(test_data)

y_pred = predictor.predict(test_data)
print(y_pred)

time.sleep(1)  # Ensure we don't use the same train directory

###############################################################
# Training custom model alongside other models using task.fit #
###############################################################

# Now we add the custom model to be trained alongside the default models:
custom_hyperparameters.update(get_hyperparameter_config('default'))
predictor = task.fit(train_data=train_data, label=label_column, hyperparameters=custom_hyperparameters)  # Train the default models plus a single default NaiveBayesModel
# predictor = task.fit(train_data=train_data, label=label_column, auto_stack=True, hyperparameters=custom_hyperparameters)  # We can even use the custom model in a multi-layer stack ensemble
predictor.leaderboard(test_data)

y_pred = predictor.predict(test_data)
print(y_pred)
