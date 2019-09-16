
import datetime
import fastparquet

from tabular.utils.loaders import load_pd
from tabular.ml.learner.default_learner import DefaultLearner
from tabular.sandbox.techops.feature_generator import FeatureGenerator
from tabular.sandbox.techops.vectorizers import vectorizer_3
import numpy as np

ID = 'ID'
LABEL = 'root_cause'

LGBM_PREDS = 'data/sandbox/techops_competition/learners/20190516_030354/models/submissions/submission_proba_20190517_005313.csv'
NN_ENSEMBLE_PREDS = 's3a://f3-green-beta-test/ashyrkou-lambdas-tests/mle-ops-it-2019/test_ensemble_v4_bag_mean_best.parquet'

TRAIN_FILE = 'training_stage_1.parquet'
TEST_FILE = 'public_test_features_stage_1.parquet'

if __name__ == '__main__':
    utcnow = datetime.datetime.utcnow()
    timestamp_str_now = utcnow.strftime("%Y%m%d_%H%M%S")
    path_prefix = 'data/sandbox/techops_competition/'
    path_train = path_prefix + TRAIN_FILE
    path_model_prefix = path_prefix + 'learners/' + timestamp_str_now + '/'
    path_test = path_prefix + TEST_FILE

    try: X = fastparquet.ParquetFile(path_train).to_pandas()
    except: X = load_pd.load(path_train)

    try: X_LGBM = fastparquet.ParquetFile(LGBM_PREDS).to_pandas()
    except: X_LGBM = load_pd.load(LGBM_PREDS)

    try: X_NN = fastparquet.ParquetFile(NN_ENSEMBLE_PREDS).to_pandas()
    except: X_NN = load_pd.load(NN_ENSEMBLE_PREDS)

    print(X_LGBM)

    print(X_NN)

    weights = [0.33, 0.67]
    preds_ensemble = [X_LGBM.values, X_NN.values]

    print(preds_ensemble)

    preds_ensemble_norm = [preds_prob * weights[i] for i, preds_prob in enumerate(preds_ensemble)]

    print(preds_ensemble_norm)

    cv_preds_ensemble = np.sum(preds_ensemble_norm, axis=0)

    print(cv_preds_ensemble)

    y_pred_proba = cv_preds_ensemble

    feature_generator = FeatureGenerator(vectorizer=vectorizer_3())
    # # feature_generator = FeatureGenerator(vectorizer=Vectorizer(max_ngrams=6, min_df=50, max_features=3000))
    learner = DefaultLearner(path_context=path_model_prefix, label=LABEL, submission_columns=[ID], feature_generator=feature_generator)
    learner.fit(
        X=X,
        # X_test=X_test,
    )
    #

    try: X_test = fastparquet.ParquetFile(path_test).to_pandas()
    except: X_test = load_pd.load(path_test)

    learner.submit_from_preds(X_test=X_test, y_pred_proba=y_pred_proba)

