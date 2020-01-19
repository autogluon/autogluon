import os
import autogluon as ag
from autogluon import ImageClassification as task
from mxnet import optimizer as optim

def task_dog_breed_identification():
    # data_loader
    data_path = '/home/ubuntu/workspace/dataset'
    dataset = 'dog-breed-identification_mini'
    dataset_path = os.path.join(data_path, dataset, 'train')
    test_path = os.path.join(data_path, dataset, 'test')
    csv_path = os.path.join(data_path, dataset, 'sample_submission.csv')
    load_dataset = task.Dataset(dataset_path)

    # images_path = os.path.join(data_path, dataset, 'images')
    # label_path =  os.path.join(data_path, dataset, 'labels.csv')
    # load_dataset = task.Dataset(images_path, label_file=label_path) # bug

    # search space
    @ag.obj(
        learning_rate=ag.space.Real(0.3, 0.5),
        momentum=ag.space.Real(0.90, 0.95),
        wd=ag.space.Real(1e-6, 1e-4, log=True),
        multi_precision=False
    )
    class NAG(optim.NAG):
        pass

    classifier = task.fit(dataset=load_dataset,
                          # pretrain model
                          # nets=ag.Categorical('resnext_101_standforddog', 'resnet151_v1d_standford'),
                          optimizer=NAG(),
                          epochs=1,
                          num_trials=2,
                          ngpus_per_trial=1,
                          batch_size=48,
                          verbose=False,
                          ensemble=2)

    multi_scale_crop = True
    if multi_scale_crop:
        scale_ratio_choice=[0.875, 0.8, 0.7] #256,280,320
        for i in scale_ratio_choice:
            inds, probs, probs_all = classifier.predict(task.Dataset(test_path), crop_ratio=i)
            ag.utils.generate_prob_csv(test_path, csv_path, probs_all, custom='./submission_%.3f.csv' % (i))
        ag.utils.generate_prob_csv(test_path, csv_path, probs_all, custom='./ensemble.csv', ensemble_list='./submission_*.csv')
    else:
        inds, probs, probs_all = classifier.predict(task.Dataset(test_path), crop_ratio=0.875)
        ag.utils.generate_prob_csv(test_path, csv_path, probs_all, custom='./submission.csv')

if __name__ == '__main__':
    task_dog_breed_identification()