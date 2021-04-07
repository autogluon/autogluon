from autogluon.vision import ImagePredictor as Task

def test_task():
    dataset, _, test_dataset = Task.Dataset.from_folders('https://autogluon.s3.amazonaws.com/datasets/shopee-iet.zip')
    model_list = Task.list_models()
    classifier = Task()
    classifier.fit(dataset, num_trials=2, hyperparameters={'epochs': 1, 'early_stop_patience': 3})
    test_result = classifier.predict(test_dataset)
    single_test = classifier.predict(test_dataset.iloc[0]['image'])
    single_proba = classifier.predict_proba(test_dataset.iloc[0]['image'])
    classifier.save('classifier.ag')
    classifier2 = Task.load('classifier.ag')
    fit_summary = classifier2.fit_summary()
    test_acc = classifier2.evaluate(test_dataset)
    test_proba = classifier2.predict_proba(test_dataset, squeeze=True)
    test_feature = classifier2.predict_feature(test_dataset)
    single_test2 = classifier2.predict(test_dataset.iloc[0]['image'])
    assert single_test2.equals(single_test)
    # to numpy
    test_proba = classifier2.predict_proba(test_dataset, as_pandas=False)
    test_feature = classifier2.predict_feature(test_dataset, as_pandas=False)
    single_test2 = classifier2.predict(test_dataset.iloc[0]['image'], as_pandas=False)
