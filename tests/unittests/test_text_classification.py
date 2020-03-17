from autogluon import TextClassification as task

def test_fit():
    dataset = task.Dataset(name='ToySST')
    predictor = task.fit(dataset, epochs=1, num_trials=1)
    print('Top-1 val acc: %.3f' % predictor.results['best_reward'])
    test_acc = predictor.evaluate(dataset)
    print('Top-1 test acc: %.3f' % test_acc)
    sentence = 'I feel this is awesome!'
    ind = predictor.predict(sentence)
    print('The input sentence sentiment is classified as [%d].' % ind.asscalar())
    print('The best configuration is:')
    print(predictor.results['best_config'])