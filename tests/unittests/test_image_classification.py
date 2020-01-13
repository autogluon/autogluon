import autogluon as ag
from autogluon import ImageClassification as task

def test_ensemble():
    dataset = task.Dataset(name='FashionMNIST')
    test_dataset = task.Dataset(name='FashionMNIST', train=False)
    classifier = task.fit(dataset,
                          epochs=10,
                          ngpus_per_trial=1,
                          verbose=False,
                          ensemble=2)
    test_acc = classifier.evaluate(test_dataset)

if __name__ == '__main__':
    test_ensemble()
