import os
import autogluon as ag
from autogluon import ImageClassification as task
from mxnet import optimizer as optim

def task_dog_breed_identification(data_path, dataset):
    images_path = os.path.join(data_path, dataset, 'images_all')
    label_path =  os.path.join(data_path, dataset, 'labels.csv')
    test_path = os.path.join(data_path, dataset, 'test')
    load_dataset = task.Dataset(images_path, label_file=label_path)

    @ag.obj(
        learning_rate=ag.space.Real(0.3, 0.5),
        momentum=ag.space.Real(0.90, 0.95),
        wd=ag.space.Real(1e-6, 1e-4, log=True),
        multi_precision=False
    )
    class NAG(optim.NAG):
        pass

    classifier = task.fit(dataset=load_dataset,
                          net=ag.Categorical('standford_dog_resnext101_64x4d', 'standford_dog_resnet152_v1'),
                          optimizer=NAG(),
                          epochs=20,
                          final_fit_epochs=180,
                          num_trials=40,
                          ngpus_per_trial=8,
                          batch_size=48,
                          verbose=False,
                          ensemble=1)

    test_dataset = task.Dataset(test_path, train=False, crop_ratio=0.65)
    inds, probs, probs_all = classifier.predict(test_dataset, set_prob_thresh=0.001)
    ag.utils.generate_prob_csv(test_dataset, probs_all, custom='./submission.csv')

if __name__ == '__main__':
    data_path = '/home/ubuntu/workspace/dataset'
    dataset = 'dog-breed-identification'
    task_dog_breed_identification(data_path, dataset)