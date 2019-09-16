
import datetime
import fastparquet

from tabular.utils.loaders import load_pd
from tabular.ml.learner.default_learner import DefaultLearner

if __name__ == '__main__':
    utcnow = datetime.datetime.utcnow()
    timestamp_str_now = utcnow.strftime("%Y%m%d_%H%M%S")
    timestamp_str = '20190426_222622'
    path_prefix = 'data/sandbox/techops_competition/'
    path_model_prefix = path_prefix + 'learners/' + timestamp_str + '/'

    learner_loaded = DefaultLearner.load(path_context=path_model_prefix)

    path_test = path_prefix + 'public_test_features_stage_1.parquet'

    try: X_test = fastparquet.ParquetFile(path_test).to_pandas()
    except: X_test = load_pd.load(path_test)

    submission = learner_loaded.predict(X_test=X_test)
