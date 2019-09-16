
from tabular.runners import predict, auto_train

ID = 'ID'
DOC_ID = 'doc_id'
LABEL = 'human_tag'


if __name__ == '__main__':
    path_prefix = 'test/data/sandbox/smoke/binary/'
    path_train = path_prefix + 'inputs/training.csv'
    path_test = path_prefix + 'inputs/test.csv'

    feature_generator_kwargs = {
        'enable_nlp_ratio_features': False,
        'enable_nlp_vectorizer_features': False,
        'enable_categorical_features': True,
        'enable_datetime_features': True,
    }

    learner_context, _ = auto_train.train(data=path_train, label=LABEL, submission_columns=[ID, DOC_ID], feature_generator_kwargs=feature_generator_kwargs, sample=None)
    y_pred = predict.predict(data=path_test, learner=learner_context)
    print(y_pred)
    print('Smoke run was successful!')
