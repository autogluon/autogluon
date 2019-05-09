import logging

import autogluon as ag
import autogluon.image_classification as task

train_dataset, valid_dataset = task.Dataset('./CIFAR10/train', './CIFAR10/valid')

models = task.fit(train_dataset)

logging.info('trials results:')
logging.info(models[0])
