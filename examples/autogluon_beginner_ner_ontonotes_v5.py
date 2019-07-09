import logging

from autogluon import named_entity_recognition as task


if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.DEBUG)
    dataset = task.Dataset(name='Ontonotes-v5',
                           train_path='/home/ubuntu/ontonotes-v5/onto.train.ner',
                           val_path='/home/ubuntu/ontonotes-v5/onto.test.ner')
    results = task.fit(dataset)

    logger.info('Best result:')
    logger.info(results.metric)
    logger.info('=========================')
    logger.info('Best search space:')
    logger.info(results.config)
