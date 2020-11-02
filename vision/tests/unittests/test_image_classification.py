from autogluon.vision import ImageClassification as task

def test_task():
    dataset, _, test_dataset = Task.Dataset.from_folders('https://autogluon.s3.amazonaws.com/datasets/shopee-iet.zip')

    classifier = Task().fit(dataset)
    test_result = classifier.predict(test_dataset)
    print('test result', test_result)
    classifier.save('classifier.ag')
    classifier2 = task.Classifier.load('classifier.ag')
    test_acc = classifier2.evaluate(test_dataset)
