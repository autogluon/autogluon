
import datetime
import fastparquet

from tabular.utils.loaders import load_pd
from tabular.ml.learner.default_learner import DefaultLearner
from tabular.feature_generators.auto_ml_feature_generator import AutoMLFeatureGenerator

ID = 'ID'
LABEL = 'subcategory_name'

SAMPLE = 10000

if __name__ == '__main__':
    utcnow = datetime.datetime.utcnow()
    timestamp_str_now = utcnow.strftime("%Y%m%d_%H%M%S")
    path_prefix = 'data/sandbox/taxonomy/'
    path_train = path_prefix + 'asins_of_interest.parquet'
    path_model_prefix = path_prefix + 'learners/' + timestamp_str_now + '/'

    try: X = fastparquet.ParquetFile(path_train).to_pandas()
    except: X = load_pd.load(path_train)

    feature_generator = AutoMLFeatureGenerator()
    learner = DefaultLearner(path_context=path_model_prefix, label=LABEL, submission_columns=[], feature_generator=feature_generator)
    learner.fit(
        X=X,
        sample=SAMPLE
    )

    # learner_loaded = Learner.load(path_context=path_model_prefix)
    #
    # path_test = path_prefix + 'public_test_features_stage_1.parquet'
    #
    # try: X_test = fastparquet.ParquetFile(path_test).to_pandas()
    # except: X_test = load_pd.load(path_test)
    #
    # submission = learner_loaded.predict(X_test=X_test)
