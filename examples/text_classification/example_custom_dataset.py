from autogluon import TextClassification as task

dataset = task.Dataset(filepath='/home/ubuntu/data/tsv/train.csv', usecols=['text', 'target'])
predictor = task.fit(dataset, epochs=1)
print('Top-1 val acc: %.3f' % predictor.results['best_reward'])
test_acc = predictor.evaluate(dataset)
print('Top-1 test acc: %.3f' % test_acc)
