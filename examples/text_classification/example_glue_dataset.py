import autogluon as ag
from autogluon import TextClassification as task

dataset = task.Dataset(name='SST')
predictor = task.fit(dataset, epochs=5, warmup_ratio=0.1, log_interval=400, seed=2, num_trials=100, dev_batch_size=8,
                     accumulate=None)
print('Top-1 val acc: %.3f' % predictor.results['best_reward'])
test_acc = predictor.evaluate(dataset)
print('Top-1 test acc: %.3f' % test_acc)
sentence = 'I feel this is awesome!'
ind = predictor.predict(sentence)
print('The input sentence sentiment is classified as [%d].' % ind.asscalar())
print('The best configuration is:')
print(predictor.results['best_config'])