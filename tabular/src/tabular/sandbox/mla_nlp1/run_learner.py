
import datetime

from tabular.utils.loaders import load_pd
from tabular.ml.learner.default_learner import DefaultLearner
from tabular.feature_generators.auto_ml_feature_generator import AutoMLFeatureGenerator

ID = 'ID'
DOC_ID = 'doc_id'
LABEL = 'human_tag'

SAMPLE = 30000

if __name__ == '__main__':
    utcnow = datetime.datetime.utcnow()
    timestamp_str_now = utcnow.strftime("%Y%m%d_%H%M%S")
    path_prefix = 'data/sandbox/mla_nlp1/'
    path_train = path_prefix + 'inputs/training.csv'
    path_model_prefix = path_prefix + 'learners/' + timestamp_str_now + '/'

    X = load_pd.load(path_train)

    feature_generator = AutoMLFeatureGenerator(enable_nlp_ratio_features=True, enable_nlp_vectorizer_features=True, enable_categorical_features=True)
    learner = DefaultLearner(path_context=path_model_prefix, label=LABEL, submission_columns=[ID, DOC_ID], feature_generator=feature_generator)
    learner.fit(
        X=X,
        sample=SAMPLE
    )
