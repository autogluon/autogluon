
import datetime

from autogluon import PredictTableColumn as task
from tabular.utils.loaders import load_pd
from tabular.feature_generators.auto_ml_feature_generator import AutoMLFeatureGenerator


# data can be string path to data or the loaded data in DataFrame format
# label is the column name of the dependent variable
# model_context [Optional] is the path of the learner's output, used to load itself later on, generated automatically if not provided
# submission columns [Optional] are a list of columns not used for training
# feature_generator_kwargs [Optional] specify params to feature generator
# sample [Optional] allows for training on only sample # of rows, for prototyping
# returns the learner model context (used to load learner back with DefaultLearner.load(model_context), and the learner itself
def train(data, label: str, X_test=None, learner_context: str = None, submission_columns: list = None, hyperparameter_tune=False, feature_generator_kwargs: dict = None, problem_type: str = None, objective_func=None):
    if type(data) == str:
        data = load_pd.load(data, encoding='latin1')
    if X_test is not None:
        if type(X_test) == str:
            X_test = load_pd.load(X_test, encoding='latin1')
    if learner_context is None:
        learner_context = generate_learner_context()
    if submission_columns is None:
        submission_columns = []
    if feature_generator_kwargs is None:
        feature_generator_kwargs = {
            'enable_nlp_ratio_features': True,
            'enable_nlp_vectorizer_features': True,
            'enable_categorical_features': True
        }

    feature_generator = AutoMLFeatureGenerator(**feature_generator_kwargs)
    learner = task.fit(train_data=data, label=label, tuning_data=X_test, output_directory=learner_context, feature_generator=feature_generator, problem_type=problem_type, objective_func=objective_func, hyperparameter_tune=hyperparameter_tune, submission_columns=submission_columns)

    return learner.path_context, learner


def generate_learner_context():
    utcnow = datetime.datetime.utcnow()
    timestamp_str_now = utcnow.strftime("%Y%m%d_%H%M%S")
    path_prefix = 'data/sandbox/automl/generic/'
    return path_prefix + 'learners/' + timestamp_str_now + '/'
