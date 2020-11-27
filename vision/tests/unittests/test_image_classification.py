from autogluon.vision import ImageClassification as Task

def test_task():
    dataset, _, test_dataset = Task.Dataset.from_folders('https://autogluon.s3.amazonaws.com/datasets/shopee-iet.zip')
    model_list = Task.list_models()
    classifier = Task()
    classifier.fit(dataset, epochs=1, num_trials=2)
    test_result = classifier.predict(test_dataset)
    print('test result', test_result)
    classifier.save('classifier.ag')
    classifier2 = Task.load('classifier.ag')
    fit_summary = classifier2.fit_summary()
    test_acc = classifier2.evaluate(test_dataset)
    test_feature = classifier2.predict_feature(test_dataset)
