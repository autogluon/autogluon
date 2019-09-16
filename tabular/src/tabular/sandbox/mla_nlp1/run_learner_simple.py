
from tabular.runners import predict, auto_train

ID = 'ID'
DOC_ID = 'doc_id'
LABEL = 'human_tag'

# Get input data here: https://drive.corp.amazon.com/folders/Grail/AutoML/data/sandbox/mla_nlp1
if __name__ == '__main__':
    path_prefix = 'data/sandbox/mla_nlp1/'
    path_train = path_prefix + 'inputs/training.csv'
    path_test = path_prefix + 'inputs/public_test_features.csv'

    learner_context, _ = auto_train.train(data=path_train, label=LABEL, submission_columns=[ID, DOC_ID], sample=5000)
    y_pred = predict.predict(data=path_test, learner=learner_context)
    print(y_pred)
