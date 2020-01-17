# import autogluon as ag
# from autogluon import ImageClassification as task
#
# """
# python benchmark.py --data-dir data --dataset dog-breed-identification --num-trials 3 --num-epochs 1 --batch-size 48 --ngpus-per-trial 1
#
# classifier = task.fit(nets=ag.Categorical('resnext_101_standforddog','resnet151_v1d_standford'),
#                       dataset=dataset,
#                       epochs=20,
#                       final_epochs=180,
#                       ngpus_per_trial=8,
#                       num_trials=40,
#                       ensemble=5)
#
# test_dataset = task.Dataset('./dog-breed-identification/test')
#
# classifier.set_prob_thresh(0.1)
# inds, probs, probs_all = classifier.predict(test_dataset, input_size=256)
# ag.utils.generate_prob_csv(test_dataset, probs_all, './submission.csv')

import os
import autogluon as ag
from autogluon import ImageClassification as task

def ensemble():

    # data_loader
    data_path = '/home/ubuntu/workspace/train_script/autogluon_git/examples/image_classification/data'
    dataset = 'dog-breed-identification'
    dataset_path = os.path.join(data_path, dataset, 'images')
    load_dataset = task.Dataset(dataset_path)
    # load_dataset = task.Dataset('dog-breed-identification/train',
    #                        label_file='dog-breed-identification/labels.csv')

    classifier = task.fit(dataset=load_dataset,
                          # nets=ag.Categorical('resnext_101_standforddog', 'resnet151_v1d_standford'),
                          epochs=1,
                          num_trials=2,
                          ngpus_per_trial=1,
                          verbose=False,
                          ensemble=2)

    # test_dataset = task.Dataset('./dog-breed-identification/test')
    test_dataset = task.Dataset(os.path.join(data_path, dataset, 'test'))
    test_acc = classifier.evaluate(test_dataset)
    print(test_acc)

    # classifier.set_prob_thresh(0.1)
    # inds, probs, probs_all = classifier.predict(test_dataset, input_size=256)
    # ag.utils.generate_prob_csv(test_dataset, probs_all, './submission.csv')


if __name__ == '__main__':
    ensemble()