import os
import autogluon as ag
from autogluon import ImageClassification as task
from mxnet import optimizer as optim

def task_dog_breed_identification(data_path, dataset):
    if dataset == 'dog-breed-identification_mini':
        dataset_path = os.path.join(data_path, dataset, 'train')
        test_path = os.path.join(data_path, dataset, 'test', 'test')
        load_dataset = task.Dataset(dataset_path)
    elif dataset == 'dog-breed-identification':
        images_path = os.path.join(data_path, dataset, 'images_all')
        label_path =  os.path.join(data_path, dataset, 'labels.csv')
        test_path = os.path.join(data_path, dataset, 'test_mini') # test_autogluon
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
                          epochs=1,
                          num_trials=1,
                          ngpus_per_trial=8,
                          batch_size=48,
                          verbose=False,
                          ensemble=1)#2

    test_dataset = task.Dataset(test_path, train=False)

    ensemble = False
    if ensemble:
        scale_ratio_choice=[0.875, 0.8, 0.7] #256,280,320
        for i in scale_ratio_choice:
            inds, probs, probs_all = classifier.predict(test_dataset, crop_ratio=i)
            ag.utils.generate_prob_csv(test_dataset, probs_all, custom='./submission_%.3f.csv' % (i))
        ag.utils.generate_prob_csv(test_dataset, probs_all, custom='./ensemble.csv', ensemble_list='./submission_*.csv')
    else:
        inds, probs, probs_all = classifier.predict(test_dataset)
        ag.utils.generate_prob_csv(test_dataset, probs_all, custom='./submission.csv', set_prob_thresh=0.1)

if __name__ == '__main__':
    data_path = '/home/ubuntu/workspace/dataset'
    dataset = 'dog-breed-identification_mini'
    # dataset = 'dog-breed-identification'
    task_dog_breed_identification(data_path, dataset)