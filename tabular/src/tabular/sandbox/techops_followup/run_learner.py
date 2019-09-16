
import datetime

from tabular.utils.loaders import load_pd
from tabular.ml.learner.default_learner import DefaultLearner
from tabular.feature_generators.auto_ml_feature_generator import AutoMLFeatureGenerator

LABEL = 'ROOT_CAUSE'

TRAIN_FILE = 'inputs/training.csv'

if __name__ == '__main__':
    utcnow = datetime.datetime.utcnow()
    timestamp_str_now = utcnow.strftime("%Y%m%d_%H%M%S")
    path_prefix = 'data/sandbox/techops_followup/'
    path_train = path_prefix + TRAIN_FILE
    path_model_prefix = path_prefix + 'learners/' + timestamp_str_now + '/'

    X = load_pd.load(path_train)

    ######
    # HACK
    print(X[LABEL])
    X[LABEL] = X[LABEL].str.split('[-]').str[0].str.strip()
    print(X[LABEL])
    ######

    feature_generator = AutoMLFeatureGenerator(enable_nlp_ratio_features=True, enable_nlp_vectorizer_features=True, enable_categorical_features=True)
    learner = DefaultLearner(path_context=path_model_prefix, label=LABEL, submission_columns=[], feature_generator=feature_generator, threshold=40)
    learner.fit(
        X=X,
    )
