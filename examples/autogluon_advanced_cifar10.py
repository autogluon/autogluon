import logging

import autogluon as ag
import autogluon.image_classification as task

train_dataset, valid_dataset = task.Dataset('./CIFAR10/train', './CIFAR10/valid')

models, best_result, search_space = task.fit(train_dataset,
                                             nets=ag.Nets([task.model_zoo.get_model('resnet18_v1'),
                                                   task.model_zoo.get_model('resnet34_v1'),
                                                   task.model_zoo.get_model('resnet50_v1'),
                                                   task.model_zoo.get_model('resnet101_v1'),
                                                   task.model_zoo.get_model('resnet152_v1')]),
                                             optimizers=ag.Optimizers([ag.optims.get_optim('sgd'),
                                                         ag.optims.get_optim('adam')]))
logging.info('trials results:')
logging.info(models)
logging.info('=========================')
logging.info('best results:')
logging.info(best_result)
logging.info('=========================')
logging.info('print search space')
logging.info(search_space)
