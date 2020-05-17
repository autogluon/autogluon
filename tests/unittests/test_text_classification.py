import os
import autogluon as ag
from autogluon import TextClassification as task

def test_fit():
    dataset = task.Dataset(name='ToySST')
    predictor = task.fit(dataset,
                         net=ag.Categorical('bert_12_768_12'),
                         pretrained_dataset=ag.Categorical('book_corpus_wiki_en_uncased'),
                         epochs=1,
                         num_trials=1,
                         batch_size=4,
                         dev_batch_size=4,
                         max_len=16,
                         ngpus_per_trial=0,
                         seed=2)
    test_acc = predictor.evaluate(dataset)
    print('accuracy is %.2f' % test_acc)
    print('finished')

def test_custom_dataset_fit():
    os.system('wget https://autogluon-hackathon.s3.amazonaws.com/demodata.zip')
    os.system('unzip -o demodata.zip')
    dataset = task.Dataset(filepath='./demodata/train.csv', usecols=['text', 'target'])
    predictor = task.fit(dataset,
                         net=ag.Categorical('bert_12_768_12'),
                         pretrained_dataset=ag.Categorical('book_corpus_wiki_en_uncased'),
                         epochs=1,
                         num_trials=1,
                         batch_size=4,
                         dev_batch_size=4,
                         max_len=16,
                         ngpus_per_trial=0,
                         seed=2)
    test_acc = predictor.evaluate(dataset)
    print('accuracy is %.2f' % test_acc)
    print('finished')
