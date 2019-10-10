import multiprocessing
from typing import AnyStr

from mxnet import gluon
from mxnet.metric import Accuracy, F1, MCC, PearsonCorrelation, CompositeEvalMetric
import gluonnlp as nlp
from gluonnlp.data import GlueCoLA, GlueSST2, GlueSTSB, GlueMRPC
from gluonnlp.data import GlueQQP, GlueRTE, GlueMNLI, GlueQNLI, GlueWNLI

from .transforms import TextDataTransform, BERTDatasetTransform

__all__ = ['transform', 'get_train_data_lengths', 'get_batchify_fn',
           'get_batch_sampler', 'get_transform_train_fn', 'get_transform_val_fn'] \
          + ['MRPCTask', 'QQPTask', 'QNLITask', 'RTETask', 'STSBTask', 'CoLATask', 'MNLITask',
             'WNLITask', 'SSTTask']

class GlueTask:
    """Abstract GLUE task class.

    Parameters
    ----------
    class_labels : list of str, or None
        Classification labels of the task.
        Set to None for regression tasks with continuous real values.
    metrics : list of EValMetric
        Evaluation metrics of the task.
    is_pair : bool
        Whether the task deals with sentence pairs or single sentences.
    label_alias : dict
        label alias dict, some different labels in dataset actually means
        the same. e.g.: {'contradictory':'contradiction'} means contradictory
        and contradiction label means the same in dataset, they will get
        the same class id.
    """
    def __init__(self, class_labels, metrics, is_pair, label_alias=None):
        self.class_labels = class_labels
        self.metrics = metrics
        self.is_pair = is_pair
        self.label_alias = label_alias

    def get_dataset(self, segment='train'):
        """Get the corresponding dataset for the task.

        Parameters
        ----------
        segment : str, default 'train'
            Dataset segments.

        Returns
        -------
        TSVDataset : the dataset of target segment.
        """
        raise NotImplementedError()

    def dataset_train(self):
        """Get the training segment of the dataset for the task.

        Returns
        -------
        tuple of str, TSVDataset : the segment name, and the dataset.
        """
        return 'train', self.get_dataset(segment='train')

    def dataset_dev(self):
        """Get the dev segment of the dataset for the task.

        Returns
        -------
        tuple of (str, TSVDataset), or list of tuple : the segment name, and the dataset.
        """
        return 'dev', self.get_dataset(segment='dev')

    def dataset_test(self):
        """Get the test segment of the dataset for the task.

        Returns
        -------
        tuple of (str, TSVDataset), or list of tuple : the segment name, and the dataset.
        """
        return 'test', self.get_dataset(segment='test')

class MRPCTask(GlueTask):
    """The MRPC task on GlueBenchmark."""
    def __init__(self):
        is_pair = True
        class_labels = ['0', '1']
        metric = CompositeEvalMetric()
        metric.add(F1())
        metric.add(Accuracy())
        super(MRPCTask, self).__init__(class_labels, metric, is_pair)

    def get_dataset(self, segment='train'):
        """Get the corresponding dataset for MRPC.

        Parameters
        ----------
        segment : str, default 'train'
            Dataset segments. Options are 'train', 'dev', 'test'.
        """
        return GlueMRPC(segment=segment)

class QQPTask(GlueTask):
    """The Quora Question Pairs task on GlueBenchmark."""
    def __init__(self):
        is_pair = True
        class_labels = ['0', '1']
        metric = CompositeEvalMetric()
        metric.add(F1())
        metric.add(Accuracy())
        super(QQPTask, self).__init__(class_labels, metric, is_pair)

    def get_dataset(self, segment='train'):
        """Get the corresponding dataset for QQP.

        Parameters
        ----------
        segment : str, default 'train'
            Dataset segments. Options are 'train', 'dev', 'test'.
        """
        return GlueQQP(segment=segment)


class RTETask(GlueTask):
    """The Recognizing Textual Entailment task on GlueBenchmark."""
    def __init__(self):
        is_pair = True
        class_labels = ['not_entailment', 'entailment']
        metric = Accuracy()
        super(RTETask, self).__init__(class_labels, metric, is_pair)

    def get_dataset(self, segment='train'):
        """Get the corresponding dataset for RTE.

        Parameters
        ----------
        segment : str, default 'train'
            Dataset segments. Options are 'train', 'dev', 'test'.
        """
        return GlueRTE(segment=segment)

class QNLITask(GlueTask):
    """The SQuAD NLI task on GlueBenchmark."""
    def __init__(self):
        is_pair = True
        class_labels = ['not_entailment', 'entailment']
        metric = Accuracy()
        super(QNLITask, self).__init__(class_labels, metric, is_pair)

    def get_dataset(self, segment='train'):
        """Get the corresponding dataset for QNLI.

        Parameters
        ----------
        segment : str, default 'train'
            Dataset segments. Options are 'train', 'dev', 'test'.
        """
        return GlueQNLI(segment=segment)

class STSBTask(GlueTask):
    """The Sentence Textual Similarity Benchmark task on GlueBenchmark."""
    def __init__(self):
        is_pair = True
        class_labels = None
        metric = PearsonCorrelation()
        super(STSBTask, self).__init__(class_labels, metric, is_pair)

    def get_dataset(self, segment='train'):
        """Get the corresponding dataset for STSB

        Parameters
        ----------
        segment : str, default 'train'
            Dataset segments. Options are 'train', 'dev', 'test'.
        """
        return GlueSTSB(segment=segment)

class CoLATask(GlueTask):
    """The Warstdadt acceptability task on GlueBenchmark."""
    def __init__(self):
        is_pair = False
        class_labels = ['0', '1']
        metric = MCC(average='micro')
        super(CoLATask, self).__init__(class_labels, metric, is_pair)

    def get_dataset(self, segment='train'):
        """Get the corresponding dataset for CoLA

        Parameters
        ----------
        segment : str, default 'train'
            Dataset segments. Options are 'train', 'dev', 'test'.
        """
        return GlueCoLA(segment=segment)

class SSTTask(GlueTask):
    """The Stanford Sentiment Treebank task on GlueBenchmark."""
    def __init__(self):
        is_pair = False
        class_labels = ['0', '1']
        self.metric = Accuracy()
        super(SSTTask, self).__init__(class_labels, self.metric, is_pair)

    def get_dataset(self, segment='train'):
        """Get the corresponding dataset for SST

        Parameters
        ----------
        segment : str, default 'train'
            Dataset segments. Options are 'train', 'dev', 'test'.
        """
        return GlueSST2(segment=segment)

class WNLITask(GlueTask):
    """The Winograd NLI task on GlueBenchmark."""
    def __init__(self):
        is_pair = True
        class_labels = ['0', '1']
        metric = Accuracy()
        super(WNLITask, self).__init__(class_labels, metric, is_pair)

    def get_dataset(self, segment='train'):
        """Get the corresponding dataset for WNLI

        Parameters
        ----------
        segment : str, default 'train'
            Dataset segments. Options are 'dev', 'test', 'train'
        """
        return GlueWNLI(segment=segment)

class MNLITask(GlueTask):
    """The Multi-Genre Natural Language Inference task on GlueBenchmark."""
    def __init__(self):
        is_pair = True
        class_labels = ['neutral', 'entailment', 'contradiction']
        metric = Accuracy()
        super(MNLITask, self).__init__(class_labels, metric, is_pair)

    def get_dataset(self, segment='train'):
        """Get the corresponding dataset for MNLI

        Parameters
        ----------
        segment : str, default 'train'
            Dataset segments. Options are 'dev_matched', 'dev_mismatched', 'test_matched',
            'test_mismatched', 'train'
        """
        return GlueMNLI(segment=segment)

    def dataset_dev(self):
        """Get the dev segment of the dataset for the task.

        Returns
        -------
        list of TSVDataset : the dataset of the dev segment.
        """
        return [('dev_matched', self.get_dataset(segment='dev_matched')),
                ('dev_mismatched', self.get_dataset(segment='dev_mismatched'))]

    def dataset_test(self):
        """Get the test segment of the dataset for the task.

        Returns
        -------
        list of TSVDataset : the dataset of the test segment.
        """
        return [('test_matched', self.get_dataset(segment='test_matched')),
                ('test_mismatched', self.get_dataset(segment='test_mismatched'))]


#
# def get_dataset(name, **kwargs):
#     """Returns a dataset by name
#
#     Parameters
#     ----------
#     name : str
#         Name of the dataset.
#
#     Returns
#     -------
#     Dataset
#         The dataset.
#     """
#     name = name.lower()
#     if name not in _dataset:
#         err_str = '"%s" is not among the following dataset list:\n\t' % (name)
#         err_str += '%s' % ('\n\t'.join(sorted(_dataset.keys())))
#         raise ValueError(err_str)
#     dataset = _dataset[name](**kwargs)
#     return dataset


def transform(dataset, transform_fn, num_workers=4):
    # The model type is necessary to pre-process it based on the inputs required to the model.
    with multiprocessing.Pool(num_workers) as pool:
        return gluon.data.SimpleDataset(pool.map(transform_fn, dataset))


def get_train_data_lengths(model_name: AnyStr, dataset, num_workers=4):
    with multiprocessing.Pool(num_workers) as pool:
        if 'bert' in model_name:
            return dataset.transform(lambda token_id, length, segment_id, label_id: length,
                                     lazy=False)
        else:
            return dataset.transform(lambda data, label: int(len(data)), lazy=False)


def get_batchify_fn(model_name: AnyStr):
    if 'bert' in model_name:
        return nlp.data.batchify.Tuple(
            nlp.data.batchify.Pad(axis=0), nlp.data.batchify.Stack(),
            nlp.data.batchify.Pad(axis=0), nlp.data.batchify.Stack(dtype='int32'))
    else:
        return nlp.data.batchify.Tuple(nlp.data.batchify.Pad(axis=0, ret_length=True),
                                       nlp.data.batchify.Stack(dtype='int32'))


def get_batch_sampler(model_name: AnyStr, train_dataset, batch_size, num_workers=4):
    train_data_lengths = get_train_data_lengths(model_name, train_dataset, num_workers)
    return nlp.data.FixedBucketSampler(train_data_lengths, batch_size=batch_size,
                                       shuffle=True,
                                       num_buckets=10, ratio=0)


def get_transform_train_fn(model_name: AnyStr, vocab: nlp.Vocab, max_sequence_length, is_pair, class_labels=None):
    if 'bert' in model_name:
        dataset_transform = BERTDatasetTransform(tokenizer=nlp.data.BERTTokenizer(vocab=vocab, lower=True),
                                              max_seq_length=max_sequence_length,
                                              pair=is_pair, class_labels=class_labels)

    else:
        dataset_transform = TextDataTransform(vocab, transforms=[
            nlp.data.ClipSequence(length=max_sequence_length)],
                                              pair=is_pair, max_sequence_length=max_sequence_length)
    return dataset_transform


def get_transform_val_fn(model_name: AnyStr, vocab: nlp.Vocab, max_sequence_length, is_pair, class_labels=None):
    return get_transform_train_fn(model_name, vocab, max_sequence_length, is_pair, class_labels)


# _dataset = {'sst': SSTTask,
#             'mnli': QNLITask,
#             'mrpc': MRPCTask,
#             'rte': RTETask,
#             }