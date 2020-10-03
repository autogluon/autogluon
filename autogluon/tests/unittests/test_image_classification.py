import autogluon as ag
from autogluon import ImageClassification as task


def test_ensemble():
    dataset = task.Dataset(name='FashionMNIST')
    test_dataset = task.Dataset(name='FashionMNIST', train=False)
    classifier = task.fit(dataset,
                          epochs=1,
                          ngpus_per_trial=1,
                          verbose=False,
                          ensemble=2)
    test_acc = classifier.evaluate(test_dataset)


def test_classifier_save_load():
    dataset = task.Dataset(name='FashionMNIST')
    test_dataset = task.Dataset(name='FashionMNIST', train=False)
    classifier = task.fit(dataset,
                          epochs=1,
                          ngpus_per_trial=1,
                          verbose=False)
    classifier.save('classifier.ag')
    classifier2 = task.Classifier.load('classifier.ag')
    test_acc = classifier2.evaluate(test_dataset)
