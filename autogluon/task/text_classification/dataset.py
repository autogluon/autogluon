import multiprocessing as mp
from typing import AnyStr
import pandas as pd

from mxnet import gluon
from mxnet.metric import Accuracy, F1, MCC, PearsonCorrelation, CompositeEvalMetric
import gluonnlp as nlp
from gluonnlp.data import GlueCoLA, GlueSST2, GlueSTSB, GlueMRPC
from gluonnlp.data import GlueQQP, GlueRTE, GlueMNLI, GlueQNLI, GlueWNLI
# from gluonnlp.data.utils import (Splitter, concat_sequence, line_splitter, whitespace_splitter)
from ...core import *
from ...utils.dataset import get_split_samplers, SampledDataset
from ...utils.tabular.utils.loaders import load_pd

__all__ = ['MRPCTask', 'QQPTask', 'QNLITask', 'RTETask', 'STSBTask', 'CoLATask', 'MNLITask',
           'WNLITask', 'SSTTask', 'AbstractGlueTask', 'get_dataset']

@func()
def get_dataset(name=None, *args, **kwargs):
    """Load a text classification dataset to train AutoGluon models on.
        
        Parameters
        ----------
        name : str
            Name describing which built-in popular text dataset to use (mostly from the GLUE NLP benchmark).
            Options include: 'mrpc', 'qqp', 'qnli', 'rte', 'sts-b', 'cola', 'mnli', 'wnli', 'sst', 'toysst'. 
            Detailed descriptions can be found in the file: `autogluon/task/text_classification/dataset.py`
    """
    path = kwargs.get('filepath', None)
    if path is not None:
        if '.tsv' or '.csv' in path:
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
        sampler, _ = get_split_samplers(dataset, split_ratio=0.2)
        return SampledDataset(dataset, sampler)

class CustomTSVClassificationTask(AbstractGlueTask):
    """

    Parameters
    ----------
    filepath: str, path of the file
        Any valid string path is acceptable.
    sep : str
        Delimiter to use.
    delimiter : str, default ``None``
        Alias for sep.
    header : int, list of int, default 'infer'
        Row number(s) to use as the column names, and the start of the
        data.
    names : array-like, optional
        List of column names to use.
    index_col : int, str, sequence of int / str, or False, default ``None``
    usecols : list-like or callable, optional
        Return a subset of the columns.
    squeeze : bool, default False
    prefix : str, optional
        Prefix to add to column numbers when no header, e.g. 'X' for X0, X1, ...
    mangle_dupe_cols : bool, default True
        Duplicate columns will be specified as 'X', 'X.1', ...'X.N', rather than
        'X'...'X'. Passing in False will cause data to be overwritten if there
        are duplicate names in the columns.
    dtype : Type name or dict of column -> type, optional
        Data type for data or columns.
    engine : {{'c', 'python'}}, optional
        Parser engine to use.
    converters : dict, optional
        Dict of functions for converting values in certain columns. Keys can either
        be integers or column labels.
    true_values : list, optional
        Values to consider as True.
    false_values : list, optional
        Values to consider as False.
    skipinitialspace : bool, default False
        Skip spaces after delimiter.
    skiprows : list-like, int or callable, optional
        Line numbers to skip (0-indexed) or number of lines to skip (int)
        at the start of the file.
        If callable, the callable function will be evaluated against the row
        indices, returning True if the row should be skipped and False otherwise.
        An example of a valid callable argument would be ``lambda x: x in [0, 2]``.
    skipfooter : int, default 0
        Number of lines at bottom of file to skip (Unsupported with engine='c').
    nrows : int, optional
        Number of rows of file to read. Useful for reading pieces of large files.
    na_values : scalar, str, list-like, or dict, optional
    keep_default_na : bool, default True
        Whether or not to include the default NaN values when parsing the data.
    na_filter : bool, default True
        Detect missing value markers (empty strings and the value of na_values).
    verbose : bool, default False
        Indicate number of NA values placed in non-numeric columns.
    skip_blank_lines : bool, default True
        If True, skip over blank lines rather than interpreting as NaN values.
    parse_dates : bool or list of int or names or list of lists or dict
    infer_datetime_format : bool, default False
    keep_date_col : bool, default False
        If True and `parse_dates` specifies combining multiple columns then
        keep the original columns.
    date_parser : function, optional
        Function to use for converting a sequence of string columns to an array of
        datetime instances.
    dayfirst : bool, default False
        DD/MM format dates, international and European format.
    cache_dates : bool, default True
        See <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html#pandas.read_csv`_ .
    iterator : bool, default False
        See <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html#pandas.read_csv`_ .
    chunksize : int, optional
        See <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html#pandas.read_csv`_ .
    compression : {{'infer', 'gzip', 'bz2', 'zip', 'xz', None}}, default 'infer'
    thousands : str, optional
        See <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html#pandas.read_csv`_ .
    decimal : str, default '.'
        See <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html#pandas.read_csv`_ .
    lineterminator : str (length 1), optional
        See <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html#pandas.read_csv`_ .
    quotechar : str (length 1), optional
        See <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html#pandas.read_csv`_ .
    quoting : int or csv.QUOTE_* instance, default 0
        See <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html#pandas.read_csv`_ .
    doublequote : bool, default ``True``
        See <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html#pandas.read_csv`_ .
    escapechar : str (length 1), optional
        See <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html#pandas.read_csv`_ .
    comment : str, optional
        See <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html#pandas.read_csv`_ .
    encoding : str, optional
        See <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html#pandas.read_csv`_ .
    dialect : str or csv.Dialect, optional
        See <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html#pandas.read_csv`_ .
    error_bad_lines : bool, default True
        See <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html#pandas.read_csv`_ .
    warn_bad_lines : bool, default True
        See <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html#pandas.read_csv`_ .
    delim_whitespace : bool, default False
        See <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html#pandas.read_csv`_ .
    low_memory : bool, default True
        See <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html#pandas.read_csv`_ .
    memory_map : bool, default False
        See <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html#pandas.read_csv`_ .
    float_precision : str, optional
        See <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html#pandas.read_csv`_ .
    """
    def __init__(self, *args, **kwargs):
        self._read(**kwargs)
        self.is_pair = False
        self.class_labels = list(set([sample[1] for sample in self.dataset]))
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
        dataset_df = pd.read_csv(**kwargs)
        dataset_df_lst = dataset_df.values.tolist()
        self.dataset = gluon.data.SimpleDataset(dataset_df_lst)
        split_ratio = kwargs.get('split_ratio', None) if 'split_ratio' in kwargs else 0.8
        self.train_sampler, self.dev_sampler = get_split_samplers(self.dataset, split_ratio=split_ratio)

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
