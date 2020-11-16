from autogluon.vision import ObjectDetection as Task

def test_task():
    dataset = Task.Dataset.from_voc('https://autogluon.s3.amazonaws.com/datasets/tiny_motorbike.zip')
    train_data, _, test_data = dataset.random_split()

    detector = Task({'epochs': 1, 'num_trials': 1, 'batch_size': 4}).fit(train_data)
    test_result = detector.predict(test_data)
    print('test result', test_result)
    detector.save('detector.ag')
    detector2 = Task.load('detector.ag')
    fit_summary = detector2.fit_summary()
    test_map = detector2.evaluate(test_data)
