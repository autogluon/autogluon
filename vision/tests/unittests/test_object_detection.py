from autogluon.vision import ObjectDetector as Task

def test_task():
    dataset = Task.Dataset.from_voc('https://autogluon.s3.amazonaws.com/datasets/tiny_motorbike.zip')
    train_data, _, test_data = dataset.random_split()

    detector = Task()
    detector.fit(train_data, num_trials=1, hyperparameters={'batch_size': 4, 'epochs': 1})
    test_result = detector.predict(test_data)
    print('test result', test_result)
    detector.save('detector.ag')
    detector2 = Task.load('detector.ag')
    fit_summary = detector2.fit_summary()
    test_map = detector2.evaluate(test_data)
    test_result2 = detector2.predict(test_data)
    assert test_result2.equals(test_result)
