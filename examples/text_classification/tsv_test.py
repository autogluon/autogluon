import autogluon as ag
from autogluon import TextClassification as task

dataset = task.Dataset(path='~/data/tsv/train.tsv', train=True, num_discard_samples=1,
                       field_separator=',', field_indices=[3,4], class_labels=['0','1'])
predictor = task.fit(dataset, epochs=5)
print('Top-1 val acc: %.3f' % predictor.results['best_reward'])