from autogluon.vision import ImagePredictor as Task
import pandas as pd
import copy

def test_task():
    dataset, _, test_dataset = Task.Dataset.from_folders('https://autogluon.s3.amazonaws.com/datasets/shopee-iet.zip')
    dataset_copy = copy.deepcopy(dataset)
    model_list = Task.list_models()
    classifier = Task()
    classifier.fit(dataset, num_trials=2, hyperparameters={'epochs': 1, 'early_stop_patience': 3})
    # assert input dataset not altered
    assert all(dataset_copy.index == dataset.index)
    test_result = classifier.predict(test_dataset)
    single_test = classifier.predict(test_dataset.iloc[0]['image'])
    single_proba = classifier.predict_proba(test_dataset.iloc[0]['image'])
    classifier.save('classifier.ag')
    classifier2 = Task.load('classifier.ag')
    fit_summary = classifier2.fit_summary()
    test_acc = classifier2.evaluate(test_dataset)
    test_proba = classifier2.predict_proba(test_dataset)
    test_feature = classifier2.predict_feature(test_dataset)
    single_test2 = classifier2.predict(test_dataset.iloc[0]['image'])
    assert isinstance(single_test2, pd.Series)
    assert single_test2.equals(single_test)
    # to numpy
    test_proba = classifier2.predict_proba(test_dataset, as_pandas=False)
    test_feature = classifier2.predict_feature(test_dataset, as_pandas=False)
    single_test2 = classifier2.predict(test_dataset.iloc[0]['image'], as_pandas=False)

def test_task_label_remap():
    ImagePredictor = Task
    train_dataset, _, test_dataset = ImagePredictor.Dataset.from_folders('https://autogluon.s3.amazonaws.com/datasets/shopee-iet.zip')
    label_remap = {0: 'd', 1: 'c', 2: 'b', 3: 'a'}
    train_dataset = train_dataset.replace({"label": label_remap})
    test_dataset = test_dataset.replace({"label": label_remap})
    predictor = ImagePredictor()
    predictor.fit(train_dataset, hyperparameters={'epochs': 2})
    pred = predictor.predict(test_dataset)
    pred_proba = predictor.predict_proba(test_dataset)
    label_remap_inverse = {col_name: i for i, col_name in enumerate(list(pred_proba.columns))}
    from autogluon.core.metrics import accuracy, log_loss
    score_accuracy = accuracy(y_true=test_dataset['label'], y_pred=pred)
    score_log_loss = log_loss(y_true=test_dataset['label'].replace(label_remap_inverse), y_pred=pred_proba.to_numpy())
    assert score_accuracy > 0.2  # relax
