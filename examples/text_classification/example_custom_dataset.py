# Step 1: Install GluonNLP + Run the dataset downloading scripts
# pip install gluonnlp==0.8.1
# sh download_dataset.sh

# Step 2: run the following example script

from autogluon import TextClassification as task

dataset = task.Dataset(filepath='./data/nlp-getting-started/train.csv', usecols=['text', 'target'])
predictor = task.fit(dataset, epochs=1)
print('Top-1 val acc: %.3f' % predictor.results['best_reward'])
test_acc = predictor.evaluate(dataset)
print('Top-1 test acc: %.3f' % test_acc)
