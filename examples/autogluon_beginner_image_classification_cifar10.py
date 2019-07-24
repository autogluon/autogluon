import logging

from autogluon import image_classification as task


if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.DEBUG)

    resources_per_trial = {
        'max_num_gpus': 1,
        'max_num_cpus': 4,
        'max_training_epochs': 3
    }
    dataset = task.Dataset(name='CIFAR10', num_workers=resources_per_trial['max_num_cpus'])
    results = task.fit(dataset, resources_per_trial=resources_per_trial)

    logger.debug('Best result:')
    logger.debug(results.metric)
    logger.debug('=========================')
    logger.debug('Best search space:')
    logger.debug(results.config)
    logger.debug('=========================')
    logger.debug('Total time cost:')
    logger.debug(results.time)
