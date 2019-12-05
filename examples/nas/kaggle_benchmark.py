import math
import autogluon as ag
from autogluon import ImageClassification as task
from autogluon import config_choice

# /Users/ysunmzn/workspace/1106darts/gluon-cv/autogluon/autogluon/task/image_classification/utils.py

# root = /your_path/sy_datasets/
root = '/home/ubuntu/workspace/1107gluoncv_cla/data/'

# dataset = 'dogs-vs-cats-redux-kernels-edition/'
# dataset = 'aerial-cactus-identification/'
# dataset = 'plant-seedlings-classification/'
# dataset = 'fisheries_Monitoring/'
# dataset = 'dog-breed-identification/'

dataset = 'shopee-iet-machine-learning-competition/'
# dataset = 'shopee-iet/'

target = config_choice(dataset, root)

classifier = task.fit(dataset = task.Dataset(target['dataset']),
                      classes = target['classes'],
                      net = target['net'],
                      optimizer = target['optimizer'],
                      lr_scheduler = target['lr_scheduler'],
                      epochs = target['epochs'],
                      ngpus_per_trial = target['ngpus_per_trial'],
                      num_trials = target['num_trials'],
                      batch_size = target['batch_size'],
                      verbose=True,
                      plot_results=True,
                      tricks = target['tricks'],
                      lr_config = target['lr_config'])# ->

from autogluon.task.image_classification.classifier import Classifier

# ？ Classifier.save(classifier.state_dict(),'dd')

print(classifier)


test_path = os.path.join('/home/ubuntu/workspace/1107gluoncv_cla/data/', dataset ,'test/fffed17d1a8e0433a934db518d7f532c.jpg')
ind, prob = classifier.predict(test_path)
print('The input picture is classified as [%s], with probability %.2f.' %
       (dataset.init().classes[ind.asscalar()], prob.asscalar()))

ag.done()
