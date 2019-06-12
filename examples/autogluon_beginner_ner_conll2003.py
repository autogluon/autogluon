import logging

from autogluon import named_entity_recognition as task


if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.DEBUG)
    indexes_format = {0: 'text', 3: 'ner'}
    dataset = task.Dataset(name='CoNLL2003',
                           train_path='/home/ubuntu/conll2003/train.txt',
                           val_path='/home/ubuntu/conll2003/test.txt',
                           indexes_format=indexes_format)
    results = task.fit(dataset)

    logger.info('Best result:')
    logger.info(results.metric)
    logger.info('=========================')
    logger.info('Best search space:')
    logger.info(results.config)
