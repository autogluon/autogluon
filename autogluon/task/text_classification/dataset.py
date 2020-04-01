import multiprocessing as mp
from typing import AnyStr
import pandas as pd

from mxnet import gluon
from mxnet.metric import Accuracy, F1, MCC, PearsonCorrelation, CompositeEvalMetric
import gluonnlp as nlp
from gluonnlp.data import GlueCoLA, GlueSST2, GlueSTSB, GlueMRPC
from gluonnlp.data import GlueQQP, GlueRTE, GlueMNLI, GlueQNLI, GlueWNLI
from ...core import *
from ...utils.dataset import get_split_samplers, SampledDataset

__all__ = ['MRPCTask', 'QQPTask', 'QNLITask', 'RTETask', 'STSBTask', 'CoLATask', 'MNLITask',
           'WNLITask', 'SSTTask', 'AbstractGlueTask', 'get_dataset', 'AbstractCustomTask']

@func()
def get_dataset(name=None, *args, **kwargs):
    """Load a text classification dataset to train AutoGluon models on.
        
        Parameters
        ----------
        name : str
            Name describing which built-in popular text dataset to use (mostly from the GLUE NLP benchmark).
            Options include: 'mrpc', 'qqp', 'qnli', 'rte', 'sts-b', 'cola', 'mnli', 'wnli', 'sst', 'toysst'. 
            Detailed descriptions can be found in the file: `autogluon/task/text_classification/dataset.py`
        **kwargs : some keyword arguments when using custom dataset as below.
        kwargs['label']: str or int Default: last column
            Index or column name of the label/target column.
    """
    path = kwargs.get('filepath', None)
    if path is not None:
        if path.endswith('.tsv') or path.endswith('.csv'):
            return CustomTSVClassificationTask(*args, **kwargs)
        else:
            raise NotImplementedError
    if name is not None and name.lower() in built_in_tasks:
        return built_in_tasks[name.lower()](*args, **kwargs)
    else:
        raise NotImplementedError

class AbstractGlueTask:
    """Abstract task class for datasets with GLUE format.

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
        raise NotImplementedError

    def dataset_train(self):
        """Get the training segment of the dataset for the task.

        Returns
        -------
        tuple of str, TSVDataset : the segment name, and the dataset.
        """
        return 'train', self.get_dataset(segment='train')

    def dataset_dev(self):
        """Get the development (i.e. validation) segment of the dataset for this task.

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

class ToySSTTask(AbstractGlueTask):
    """The Stanford Sentiment Treebank task on GLUE benchmark."""
    def __init__(self):
        is_pair = False
        class_labels = ['0', '1']
        self.metric = Accuracy()
        super(ToySSTTask, self).__init__(class_labels, self.metric, is_pair)

    def get_dataset(self, segment='train'):
        """Get the corresponding dataset for SST

        Parameters
        ----------
        segment : str, default 'train'
            Dataset segments. Options are 'train', 'dev', 'test'.
        """
        dataset = GlueSST2(segment=segment)
        sampler, _ = get_split_samplers(dataset, split_ratio=0.01)
        return SampledDataset(dataset, sampler)

class AbstractCustomTask:
    """Abstract task class for custom datasets to be used with BERT-family models.

    Parameters
    ----------
    class_labels : list of str, or None
        Classification labels of the task.
        Set to None for regression tasks with continuous real values (TODO).
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
        Dataset : the dataset of target segment.
        """
        raise NotImplementedError

    def dataset_train(self):
        """Get the training segment of the dataset for the task.

        Returns
        -------
        tuple of str, Dataset : the segment name, and the dataset.
        """
        return 'train', self.get_dataset(segment='train')

    def dataset_dev(self):
        """Get the development (i.e. validation) segment of the dataset for this task.

        Returns
        -------
        tuple of str, Dataset : the segment name, and the dataset.
        """
        return 'dev', self.get_dataset(segment='dev')

    def dataset_test(self):
        """Get the test segment of the dataset for the task.

        Returns
        -------
        tuple of str, Dataset : the segment name, and the dataset.
        """
        return 'test', self.get_dataset(segment='test')


class CustomTSVClassificationTask(AbstractCustomTask):
    """

    Parameters
    ----------
    filepath: str, path of the file
        Any valid string path is acceptable.
    sep : str
        Delimiter to use.
    header : int, list of int, default 'infer'
        Row number(s) to use as the column names, and the start of the
        data.
    names : array-like, optional
        List of column names to use.
    usecols : list-like or callable, optional
        Return a subset of the columns.
    For other keyword arguments, please see
        <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html#pandas.read_csv`_ .
    """
    def __init__(self, *args, **kwargs):
        self._read(**kwargs)
        self.is_pair = False
        self._label_column_id = self._index_label_column(**kwargs)
        self.class_labels = list(set([sample[self._label_column_id] for sample in self.dataset]))
        self.metric = Accuracy()
        super(CustomTSVClassificationTask, self).__init__(self.class_labels, self.metric, self.is_pair)

    def get_dataset(self, segment='train'):
        """Get the corresponding dataset for the task.

        Parameters
        ----------
        segment : str, default 'train'
            Dataset segments.

        Returns
        -------
        The dataset of target segment.
        """
        if segment == 'train':
            return SampledDataset(self.dataset, self.train_sampler)
        elif segment == 'dev':
            return SampledDataset(self.dataset, self.dev_sampler)
        else:
            raise NotImplementedError

    def _read(self, **kwargs):
        path = kwargs.get('filepath', None)
        kwargs['filepath_or_buffer'] = path
        kwargs.pop("filepath")
        low_memory = kwargs.get('low_memory', None)
        if low_memory is None:
            kwargs['low_memory'] = False
        dataset_df = pd.read_csv(**kwargs)
        dataset_df_lst = dataset_df.values.tolist()
        self.dataset = gluon.data.SimpleDataset(dataset_df_lst)
        split_ratio = kwargs.get('split_ratio', None) if 'split_ratio' in kwargs else 0.8
        self.train_sampler, self.dev_sampler = get_split_samplers(self.dataset, split_ratio=split_ratio)

    def _index_label_column(self, **kwargs):
        label_column_id = kwargs.get('label', None)
        if label_column_id is None:
            return len(self.dataset[0]) - 1
        if isinstance(label_column_id, str):
            usecols = kwargs.get('usecols', None)
            if usecols is None:
                raise NotImplementedError
            else:
                for i, col in enumerate(usecols):
                    if col == label_column_id:
                        return i
        elif isinstance(label_column_id, int):
            return label_column_id
        else:
            raise NotImplementedError

    def dataset_train(self):
        """Get the training segment of the dataset for the task.

        Returns
        -------
        tuple of str, Dataset : the segment name, and the dataset.
        """
        return 'train', self.get_dataset('train')

    def dataset_dev(self):
        """Get the dev segment of the dataset for the task.

        Returns
        -------
        tuple of str, Dataset : the segment name, and the dataset.
        """
        return 'dev', self.get_dataset('dev')

class MRPCTask(AbstractGlueTask):
    """The MRPC task on GLUE benchmark."""
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

class QQPTask(AbstractGlueTask):
    """The Quora Question Pairs task on GLUE benchmark."""
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


class RTETask(AbstractGlueTask):
    """The Recognizing Textual Entailment task on GLUE benchmark."""
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

class QNLITask(AbstractGlueTask):
    """The SQuAD NLI task on GLUE benchmark."""
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

class STSBTask(AbstractGlueTask):
    """The Sentence Textual Similarity Benchmark task on GLUE benchmark."""
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

class CoLATask(AbstractGlueTask):
    """The Warstdadt acceptability task on GLUE benchmark."""
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

class SSTTask(AbstractGlueTask):
    """The Stanford Sentiment Treebank task on GLUE benchmark."""
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

class WNLITask(AbstractGlueTask):
    """The Winograd NLI task on GLUE benchmark."""
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

class MNLITask(AbstractGlueTask):
    """The Multi-Genre Natural Language Inference task on GLUE benchmark."""
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

built_in_tasks = {
    'mrpc': MRPCTask,
    'qqp': QQPTask,
    'qnli': QNLITask,
    'rte': RTETask,
    'sts-b': STSBTask,
    'cola': CoLATask,
    'mnli': MNLITask,
    'wnli': WNLITask,
    'sst': SSTTask,
    'toysst': ToySSTTask,
}
