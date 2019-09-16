
import datetime

from tabular.ml.learner.default_learner import DefaultLearner

if __name__ == '__main__':
    utcnow = datetime.datetime.utcnow()
    timestamp_str = '20190429_210050'
    path_prefix = 'data/sandbox/techops_competition/'
    path_model_prefix = path_prefix + 'learners/' + timestamp_str + '/'

    learner_loaded = DefaultLearner.load(path_context=path_model_prefix)

    learner_loaded.evaluate()
