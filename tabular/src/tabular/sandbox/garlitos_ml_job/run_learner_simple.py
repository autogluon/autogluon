
from tabular.runners import predict, auto_train
from tabular.ml.learner.default_learner import DefaultLearner
from f3_grail_data_frame_utilities.loaders import load_pd



if __name__ == '__main__':
    path_prefix = 'data/sandbox/garlitos_ml_job/'
    path_train = path_prefix + 'inputs/training.csv'
    path_test = path_prefix + 'inputs/testing.csv'
    label = 'Actual Sales'

    learner_context, _ = auto_train.train(data=path_train, label=label, submission_columns=[], problem_type='regression')

    learner = DefaultLearner.load(path_context=learner_context)
    X_test = load_pd.load(path_test, encoding='latin1')

    score = learner.score(X_test)

    print('score:')
    print(score)
