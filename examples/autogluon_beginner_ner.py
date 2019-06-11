import logging

from autogluon import named_entity_recognition as task


if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.DEBUG)

    dataset = task.Dataset(name='CoNLL2003',
                           train_path='/Users/karjar/Downloads/conll2003/train.txt',
                           val_path='/Users/karjar/Downloads/conll2003/valid.txt')
    print()

