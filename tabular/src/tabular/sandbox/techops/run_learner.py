
import datetime

from tabular.utils.loaders import load_pd
from tabular.ml.learner.default_learner import DefaultLearner
from tabular.sandbox.techops.feature_generator import FeatureGenerator
from tabular.sandbox.techops.vectorizers import vectorizer_3

ID = 'ID'
LABEL = 'root_cause'

TRAIN_FILE = 'training_stage_1.parquet'
TEST_FILE = 'public_test_features_stage_1.parquet'

if __name__ == '__main__':
    utcnow = datetime.datetime.utcnow()
    timestamp_str_now = utcnow.strftime("%Y%m%d_%H%M%S")
    path_prefix = 'data/sandbox/techops_competition/'
    path_train = path_prefix + TRAIN_FILE
    path_model_prefix = path_prefix + 'learners/' + timestamp_str_now + '/'
    path_test = path_prefix + TEST_FILE

    X = load_pd.load(path_train)

    feature_generator = FeatureGenerator(vectorizer=vectorizer_3())
    learner = DefaultLearner(path_context=path_model_prefix, label=LABEL, submission_columns=[ID], feature_generator=feature_generator)
    learner.fit(
        X=X,
    )

    X_test = load_pd.load(path_test)

    learner_loaded = DefaultLearner.load(path_context=path_model_prefix)

    submission = learner_loaded.predict_and_submit(X_test=X_test)
