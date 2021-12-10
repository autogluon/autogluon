import logging
from collections import OrderedDict
import numpy as np
import pandas as pd
import mxnet as mx

from autogluon.common.utils.multiprocessing_utils import is_fork_enabled, is_forkserver_enabled
from autogluon.core.utils.loaders import load_pkl
from autogluon.core.utils.savers import save_pkl
from autogluon.core.constants import BINARY, MULTICLASS, REGRESSION, SOFTCLASS

logger = logging.getLogger(__name__) # TODO: Currently unused


class TabularNNDataset:
    """ Class for preprocessing & storing/feeding data batches used by tabular data neural networks. Assumes entire dataset can be loaded into numpy arrays.
        Original Data table may contain numerical, categorical, and text (language) fields.

        Attributes:
            dataset (mxnet.gluon.data.dataset): Contains the raw data (use dataset._data to access).
                                                Different indices in this list correspond to different types of inputs to the neural network (each is 2D ND array)
                                                All vector-valued (continuous & one-hot) features are concatenated together into a single index of the dataset.
            data_desc (list[str]): Describes the data type of each index of dataset (options: 'vector','embed_<featname>', 'language_<featname>')
            dataloader (mxnet.gluon.data.DataLoader): Loads batches of data from dataset for neural net training and inference.
            embed_indices (list): which columns in dataset correspond to embed features (order matters!)
            language_indices (list): which columns in dataset correspond to language features (order matters!)
            vecfeature_col_map (dict): maps vector_feature_name ->  columns of dataset._data[vector] array that contain the data for this feature
            feature_dataindex_map (dict): maps feature_name -> i such that dataset._data[i] = data array for this feature. Cannot be used for vector-valued features, instead use vecfeature_col_map
            feature_groups (dict): maps feature_type (ie. 'vector' or 'embed' or 'language') to list of feature names of this type (empty list if there are no features of this type)
            vectordata_index (int): describes which element of the dataset._data list holds the vector data matrix (access via self.dataset._data[self.vectordata_index]); None if no vector features
            label_index (int): describing which element of the dataset._data list holds labels (access via self.dataset._data[self.label_index].asnumpy()); None if no labels
            num_categories_per_embedfeature (list): Number of categories for each embedding feature (order matters!)
            num_examples (int): number of examples in this dataset
            num_features (int): number of features (we only consider original variables as features, so num_features may not correspond to dimensionality of the data eg in the case of one-hot encoding)
            num_classes (int): number of classes (only used for multiclass classification)

        Note: Default numerical data-type is converted to float32 (as well as labels in regression).
    """

    DATAOBJ_SUFFIX = '_tabNNdataset.pkl' # hard-coded names for files. This file contains pickled TabularNNDataset object
    DATAVALUES_SUFFIX = '_tabNNdata.npz' # This file contains raw data values as data_list of NDArrays

    def __init__(self, processed_array, feature_arraycol_map, feature_type_map, batch_size, num_dataloading_workers, problem_type,
                 labels=None, is_test=True):
        """ Args:
                processed_array: 2D numpy array returned by preprocessor. Contains raw data of all features as columns
                feature_arraycol_map (OrderedDict): Mapsfeature-name -> list of column-indices in processed_array corresponding to this feature
                feature_type_map (OrderedDict): Maps feature-name -> feature_type string (options: 'vector', 'embed', 'language')
                labels (pd.Series): list of labels (y) if available
                batch_size (int): number of examples to put in each mini-batch
                num_dataloading_workers (int): number of threads to devote to loading mini-batches of data rather than model-training
        """
        self.dataset = None
        self.dataloader = None
        self.problem_type = problem_type
        self.num_examples = processed_array.shape[0]
        self.num_features = len(feature_arraycol_map) # number of features (!=dim(processed_array) because some features may be vector-valued, eg one-hot)
        self.batch_size = min(self.num_examples, batch_size)
        self.is_test = is_test
        self.num_dataloading_workers = num_dataloading_workers
        last_batch_size = self.num_examples % self.batch_size
        if last_batch_size == 0:
            last_batch_size = self.batch_size
        # TODO: The code fixes the crash on mxnet gluon interpreting a single value in a batch incorrectly.
        #  Comment out to see crash if data would have single row as final batch on test prediction (such as 1025 rows for batch size 512)
        if (self.num_examples != 1) and self.is_test and (last_batch_size == 1):
            init_batch_size = self.batch_size
            while last_batch_size == 1:
                self.batch_size = self.batch_size + 1
                last_batch_size = self.num_examples % self.batch_size
                if last_batch_size == 0:
                    last_batch_size = self.batch_size
                if self.batch_size > init_batch_size+10:
                    # Hard set to avoid potential infinite loop, don't think its mathematically possible to reach this code however.
                    self.batch_size = self.num_examples
                    last_batch_size = 0

        if feature_arraycol_map.keys() != feature_type_map.keys():
            raise ValueError("feature_arraycol_map and feature_type_map must share same keys")
        self.feature_groups = {'vector': [], 'embed': [], 'language': []} # maps feature_type -> list of feature_names (order is preserved in list)
        self.feature_type_map = feature_type_map
        for feature in feature_type_map:
            if feature_type_map[feature] == 'vector':
                self.feature_groups['vector'].append(feature)
            elif feature_type_map[feature] == 'embed':
                self.feature_groups['embed'].append(feature)
            elif feature_type_map[feature] == 'language':
                self.feature_groups['language'].append(feature)
            else:
                raise ValueError("unknown feature type: %s" % feature)

        if not self.is_test and labels is None:
            raise ValueError("labels must be provided when is_test = False")
        if labels is not None and len(labels) != self.num_examples:
            raise ValueError("number of labels and training examples do not match")

        data_list = [] # stores all data of each feature-type in list used to construct MXNet dataset. Each index of list = 2D NDArray.
        self.label_index = None # int describing which element of the dataset._data list holds labels
        self.data_desc = [] # describes feature-type of each index of data_list
        self.vectordata_index = None # int describing which element of the dataset._data list holds the vector data matrix
        self.vecfeature_col_map = {} # maps vector_feature_name ->  columns of dataset._data[vector] array that contain data for this feature
        self.feature_dataindex_map = {} # maps feature_name -> i such that dataset._data[i] = data array for this feature. Cannot be used for vector-valued features, instead use: self.vecfeature_col_map

        if len(self.feature_groups['vector']) > 0:
            vector_inds = [] # columns of processed_array corresponding to vector data
            for feature in feature_type_map:
                if feature_type_map[feature] == 'vector':
                    current_last_ind = len(vector_inds) # current last index of the vector datamatrix
                    vector_inds += feature_arraycol_map[feature]
                    new_last_ind = len(vector_inds) # new last index of the vector datamatrix
                    self.vecfeature_col_map[feature] = list(range(current_last_ind, new_last_ind))
            data_list.append(mx.nd.array(processed_array[:,vector_inds], dtype='float32')) # Matrix of data from all vector features
            self.data_desc.append("vector")
            self.vectordata_index = len(data_list) - 1

        if len(self.feature_groups['embed']) > 0:
            for feature in feature_type_map:
                if feature_type_map[feature] == 'embed':
                    feature_colind = feature_arraycol_map[feature]
                    data_list.append(mx.nd.array(processed_array[:,feature_colind], dtype='int32')) # array of ints with data for this embedding feature
                    self.data_desc.append("embed")
                    self.feature_dataindex_map[feature]  = len(data_list)-1

        if len(self.feature_groups['language']) > 0:
            for feature in feature_type_map:
                if feature_type_map[feature] == 'language':
                    feature_colinds = feature_arraycol_map[feature]
                    data_list.append(mx.nd.array(processed_array[:,feature_colinds], dtype='int32')) # array of ints with data for this language feature
                    self.data_desc.append("language")
                    self.feature_dataindex_map[feature]  = len(data_list)-1

        self.num_classes = None
        if labels is not None:
            labels = np.array(labels)
            self.data_desc.append("label")
            self.label_index = len(data_list) # To access data labels, use: self.dataset._data[self.label_index]
            self.num_classes = None
            if self.problem_type == SOFTCLASS:
                self.num_classes = labels.shape[1]
                data_list.append(mx.nd.array(labels))
            else:
                if self.problem_type == REGRESSION and labels.dtype != np.float32:
                    labels = labels.astype('float32') # Convert to proper float-type if not already
                elif self.problem_type in [BINARY, MULTICLASS]:
                    self.num_classes = len(set(labels))
                data_list.append(mx.nd.array(labels.reshape(len(labels),1)))

        self.embed_indices = [i for i in range(len(self.data_desc)) if 'embed' in self.data_desc[i]] # list of indices of embedding features in self.dataset, order matters!
        self.language_indices = [i for i in range(len(self.data_desc)) if 'language' in self.data_desc[i]]  # list of indices of language features in self.dataset, order matters!
        self.num_categories_per_embed_feature = None
        self.generate_dataset_and_dataloader(data_list=data_list)
        if not self.is_test:
            self.num_categories_per_embedfeature = self.getNumCategoriesEmbeddings()

    def generate_dataset_and_dataloader(self, data_list):
        self.dataset = mx.gluon.data.dataset.ArrayDataset(*data_list)  # Access ith embedding-feature via: self.dataset._data[self.data_desc.index('embed_'+str(i))].asnumpy()
        self.dataloader = mx.gluon.data.DataLoader(
            self.dataset, self.batch_size, shuffle=not self.is_test,
            last_batch='keep' if self.is_test else 'rollover',

           # local thread version is faster unless fork is enabled
           num_workers=self.num_dataloading_workers if is_fork_enabled() else 0,

           # need to use threadpool if forkserver is enabled, otherwise GIL will be locked
            # please note: this will make training slower
           thread_pool=is_forkserver_enabled(),
        )  # no need to shuffle test data

    def has_vector_features(self):
        """ Returns boolean indicating whether this dataset contains vector features """
        return self.vectordata_index is not None

    def num_embed_features(self):
        """ Returns number of embed features in this dataset """
        return len(self.feature_groups['embed'])

    def num_language_features(self):
        """ Returns number of language features in this dataset """
        return len(self.feature_groups['language'])

    def num_vector_features(self):
        """ Number of vector features (each onehot feature counts = 1, regardless of how many categories) """
        return len(self.feature_groups['vector'])

    def get_labels(self):
        """ Returns numpy array of labels for this dataset """
        if self.label_index is not None:
            if self.problem_type == SOFTCLASS:
                return self.dataset._data[self.label_index].asnumpy()
            else:
                return self.dataset._data[self.label_index].asnumpy().flatten()
        else:
            return None

    def getNumCategoriesEmbeddings(self):
        """ Returns number of categories for each embedding feature.
            Should only be applied to training data.
            If training data feature contains unique levels 1,...,n-1, there are actually n categories,
            since category n is reserved for unknown test-time categories.
        """
        if self.num_categories_per_embed_feature is not None:
            return self.num_categories_per_embedfeature
        else:
            num_embed_feats = self.num_embed_features()
            num_categories_per_embedfeature = [0] * num_embed_feats
            for i in range(num_embed_feats):
                feat_i = self.feature_groups['embed'][i]
                feat_i_data = self.get_feature_data(feat_i).flatten().tolist()
                num_categories_i = len(set(feat_i_data)) # number of categories for ith feature
                num_categories_per_embedfeature[i] = num_categories_i + 1 # to account for unknown test-time categories
            return num_categories_per_embedfeature

    def get_feature_data(self, feature, asnumpy=True):
        """ Returns all data for this feature.
            Args:
                feature (str): name of feature of interest (in processed dataframe)
                asnumpy (bool): should we return 2D numpy array or MXNet NDarray
        """
        nonvector_featuretypes = set(['embed', 'language'])
        if feature not in self.feature_type_map:
            raise ValueError("unknown feature encountered: %s" % feature)
        if self.feature_type_map[feature] == 'vector':
            vector_datamatrix = self.dataset._data[self.vectordata_index] # does not work for one-hot...
            feature_data = vector_datamatrix[:, self.vecfeature_col_map[feature]]
        elif self.feature_type_map[feature] in nonvector_featuretypes:
            feature_idx = self.feature_dataindex_map[feature]
            feature_data = self.dataset._data[feature_idx]
        else:
            raise ValueError("Unknown feature specified: " % feature)
        if asnumpy:
            return feature_data.asnumpy()
        else:
            return feature_data

    def get_feature_batch(self, feature, data_batch, asnumpy=False):
        """ Returns part of this batch corresponding to data from a single feature
            Args:
                data_batch (nd.array): the batch of data as provided by self.dataloader
            Returns:

        """
        nonvector_featuretypes = set(['embed', 'language'])
        if feature not in self.feature_type_map:
            raise ValueError("unknown feature encountered: %s" % feature)
        if self.feature_type_map[feature] == 'vector':
            vector_datamatrix = data_batch[self.vectordata_index]
            feature_data = vector_datamatrix[:, self.vecfeature_col_map[feature]]
        elif self.feature_type_map[feature] in nonvector_featuretypes:
            feature_idx = self.feature_dataindex_map[feature]
            feature_data = data_batch[feature_idx]
        else:
            raise ValueError("Unknown feature specified: " % feature)
        if asnumpy:
            return feature_data.asnumpy()
        else:
            return feature_data

    def format_batch_data(self, data_batch, ctx):
        """ Partitions data from this batch into different data types.
            Args:
                data_batch (nd.array): the batch of data as provided by self.dataloader
            Returns:
                formatted_batch (dict): {'vector': array of vector_datamatrix,
                                         'embed': list of embedding features' batch data,
                                         'language': list of language features batch data,
                                         'label': array of labels}
                                        where each key in dict may be missing.
        """
        if not isinstance(data_batch, list):
            data_batch = [data_batch] # Need to convert to list if dimension was dropped during batching

        if len(data_batch[0].shape) == 1:
            data_batch[0] = data_batch[0].expand_dims(axis=0)
        formatted_batch = {}
        if self.has_vector_features(): # None if there is no vector data
            formatted_batch['vector'] = data_batch[self.vectordata_index].as_in_context(ctx)
        if self.num_embed_features() > 0:
            formatted_batch['embed'] = []
            for i in self.embed_indices:
                formatted_batch['embed'].append(data_batch[i].as_in_context(ctx))
        if self.num_language_features() > 0:
            formatted_batch['language'] = []
            for i in self.language_indices:
                formatted_batch['language'].append(data_batch[i].as_in_context(ctx))
        if self.label_index is not None: # is None if there are no labels
            formatted_batch['label'] = data_batch[self.label_index].as_in_context(ctx)

        return formatted_batch

    def mask_features_batch(self, features, mask_value, data_batch):
        """ Returns new batch where all values of the indicated features have been replaced by the provided mask_value.
            Args:
                features (list[str]): list of feature names that should be masked.
                mask_value (float): value of mask which original feature values should be replaced by. If None, we replace by mean/mode/unknown
                data_batch (nd.array): the batch of data as provided by self.dataloader
            Returns:
                new_batch (nd.array): batch of masked data in same format as data_batch
        """
        return None # TODO

    def save(self, file_prefix=""):
        """ Additional naming changes will be appended to end of file_prefix (must contain full absolute path) """
        dataobj_file = file_prefix + self.DATAOBJ_SUFFIX
        datalist_file = file_prefix + self.DATAVALUES_SUFFIX
        data_list = self.dataset._data
        self.dataset = None  # Avoid pickling these
        self.dataloader = None
        save_pkl.save(path=dataobj_file, object=self)
        mx.nd.save(datalist_file, data_list)
        logger.debug("TabularNN Dataset saved to files: \n %s \n %s" % (dataobj_file, datalist_file))

    @classmethod
    def load(cls, file_prefix=""):
        """ Additional naming changes will be appended to end of file_prefix (must contain full absolute path) """
        dataobj_file = file_prefix + cls.DATAOBJ_SUFFIX
        datalist_file = file_prefix + cls.DATAVALUES_SUFFIX
        dataset: TabularNNDataset = load_pkl.load(path=dataobj_file)
        data_list = mx.nd.load(datalist_file)
        dataset.generate_dataset_and_dataloader(data_list=data_list)
        logger.debug("TabularNN Dataset loaded from files: \n %s \n %s" % (dataobj_file, datalist_file))
        return dataset
