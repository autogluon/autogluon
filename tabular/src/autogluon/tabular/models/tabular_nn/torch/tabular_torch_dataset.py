import os
import logging

import torch
import numpy as np

from autogluon.core.constants import BINARY, MULTICLASS, REGRESSION, SOFTCLASS, QUANTILE

logger = logging.getLogger(__name__)


class TabularTorchDataset(torch.utils.data.Dataset):
    """
        This class follows the structure of TabularNNDataset in tabular_nn_dataset.py (using Pytorch DataLoader instead of MXNet DataLoader),

        Class for preprocessing & storing/feeding data batches used by pytorch neural networks for tabular data.
        Assumes entire dataset can be loaded into numpy arrays.
        Original data table may contain numeric and categorical fields and missing values.

        Attributes:
            data_list (list[np.array]): Contains the raw data. Different indices in this list correspond to different
                                        types of inputs to the neural network (each is 2D array). All vector-valued
                                        (continuous & one-hot) features are concatenated together into a single index
                                        of the dataset.
            data_desc (list[str]): Describes the data type of each index of dataset
                                   (options: 'vector','embed_<featname>')
            embed_indices (list): which columns in dataset correspond to embed features (order matters!)
            vecfeature_col_map (dict): maps vector_feature_name ->  columns of dataset._data[vector] array that
                                       contain the data for this feature
            feature_dataindex_map (dict): maps feature_name -> i such that dataset._data[i] = data array for
                                          this feature. Cannot be used for vector-valued features,
                                          instead use vecfeature_col_map
            feature_groups (dict): maps feature_type (ie. 'vector' or 'embed') to list of feature
                                   names of this type (empty list if there are no features of this type)
            vectordata_index (int): describes which element of the dataset._data list holds the vector data matrix
                                    (access via self.data_list[self.vectordata_index]); None if no vector features
            label_index (int): describing which element of the dataset._data list holds labels
                               (access via self.data_list[self.label_index]); None if no labels
            num_categories_per_embedfeature (list): Number of categories for each embedding feature (order matters!)
            num_examples (int): number of examples in this dataset
            num_features (int): number of features (we only consider original variables as features, so num_features
                                may not correspond to dimensionality of the data eg in the case of one-hot encoding)
        Note: Default numerical data-type is converted to float32.
    """
    # hard-coded names for files. This file contains pickled torch.util.data.Dataset object
    DATAOBJ_SUFFIX = '_tabdataset_torch.pt'

    def __init__(self, processed_array, feature_arraycol_map, feature_type_map, problem_type, labels=None):
        """ Args:
            processed_array: 2D numpy array returned by preprocessor. Contains raw data of all features as columns
            feature_arraycol_map (OrderedDict): Mapsfeature-name -> list of column-indices in processed_array
                                                corresponding to this feature
            feature_type_map (OrderedDict): Maps feature-name -> feature_type string
                                            (options: 'vector', 'embed')
            problem_type (str): what prediction task this data is used for.
            labels (pd.Series): list of labels (y) if available
        """
        self.problem_type = problem_type
        self.num_examples = processed_array.shape[0]
        self.num_features = len(feature_arraycol_map)
        if feature_arraycol_map.keys() != feature_type_map.keys():
            raise ValueError("feature_arraycol_map and feature_type_map must share same keys")
        self.feature_groups = {'vector': [], 'embed': []}
        self.feature_type_map = feature_type_map
        for feature in feature_type_map:
            if feature_type_map[feature] == 'vector':
                self.feature_groups['vector'].append(feature)
            elif feature_type_map[feature] == 'embed':
                self.feature_groups['embed'].append(feature)
            else:
                raise ValueError("unknown feature type: %s" % feature)
        if labels is not None and len(labels) != self.num_examples:
            raise ValueError("number of labels and training examples do not match")

        self.data_desc = []
        self.data_list = []
        self.label_index = None
        self.vectordata_index = None
        self.vecfeature_col_map = {}
        self.feature_dataindex_map = {}
        self.num_classes = None

        # numerical data
        if len(self.feature_groups['vector']) > 0:
            vector_inds = []
            for feature in feature_type_map:
                if feature_type_map[feature] == 'vector':
                    current_last_ind = len(vector_inds)
                    vector_inds += feature_arraycol_map[feature]
                    new_last_ind = len(vector_inds)
                    self.vecfeature_col_map[feature] = list(range(current_last_ind, new_last_ind))
            self.data_list.append(processed_array[:, vector_inds].astype('float32'))
            self.data_desc.append('vector')
            self.vectordata_index = len(self.data_list) - 1

        # embedding data
        if len(self.feature_groups['embed']) > 0:
            for feature in feature_type_map:
                if feature_type_map[feature] == 'embed':
                    feature_colind = feature_arraycol_map[feature]
                    self.data_list.append(processed_array[:, feature_colind].astype('int64').flatten())
                    self.data_desc.append('embed')
                    self.feature_dataindex_map[feature] = len(self.data_list) - 1

        # output (target) data
        if labels is not None:
            labels = np.array(labels)
            self.data_desc.append("label")
            self.label_index = len(self.data_list)
            if self.problem_type == SOFTCLASS:
                self.num_classes = labels.shape[1]
                self.data_list.append(labels.astype('float32'))
            else:
                if self.problem_type in [REGRESSION, QUANTILE] and labels.dtype != np.float32:
                    labels = labels.astype('float32')  # Convert to proper float-type if not already
                elif self.problem_type in [BINARY, MULTICLASS]:
                    self.num_classes = len(set(labels))
                    labels = labels.astype('long')
                self.data_list.append(labels.reshape(-1, 1))

        self.embed_indices = [i for i in range(len(self.data_desc)) if 'embed' in self.data_desc[i]]
        self.num_categories_per_embed_feature = None
        self.num_categories_per_embedfeature = self.getNumCategoriesEmbeddings()

    def __getitem__(self, idx):
        output_dict = {}
        if self.has_vector_features():
            output_dict['vector'] = self.data_list[self.vectordata_index][idx]
        if self.num_embed_features() > 0:
            output_dict['embed'] = []
            for i in self.embed_indices:
                output_dict['embed'].append(self.data_list[i][idx])
        if self.label_index is not None:
            output_dict['label'] = self.data_list[self.label_index][idx]
        return output_dict

    def __len__(self):
        return self.num_examples

    def has_vector_features(self):
        """ Returns boolean indicating whether this dataset contains vector features """
        return self.vectordata_index is not None

    def num_embed_features(self):
        """ Returns number of embed features in this dataset """
        return len(self.feature_groups['embed'])

    def num_vector_features(self):
        """ Number of vector features (each onehot feature counts = 1, regardless of how many categories) """
        return len(self.feature_groups['vector'])

    def get_labels(self):
        """ Returns numpy array of labels for this dataset """
        if self.label_index is not None:
            return self.data_list[self.label_index]
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

    def get_feature_data(self, feature):
        """ Returns all data for this feature.
            Args:
                feature (str): name of feature of interest (in processed dataframe)
        """
        nonvector_featuretypes = set(['embed'])
        if feature not in self.feature_type_map:
            raise ValueError("unknown feature encountered: %s" % feature)
        if self.feature_type_map[feature] == 'vector':
            vector_datamatrix = self.data_list[self.vectordata_index]
            feature_data = vector_datamatrix[:, self.vecfeature_col_map[feature]]
        elif self.feature_type_map[feature] in nonvector_featuretypes:
            feature_idx = self.feature_dataindex_map[feature]
            feature_data = self.data_list[feature_idx]
        else:
            raise ValueError("Unknown feature specified: " % feature)
        return feature_data

    def save(self, file_prefix=""):
        """ Additional naming changes will be appended to end of file_prefix (must contain full absolute path) """
        dataobj_file = file_prefix + self.DATAOBJ_SUFFIX
        if not os.path.exists(os.path.dirname(dataobj_file)):
            os.makedirs(os.path.dirname(dataobj_file))
        torch.save(self, dataobj_file)
        logger.debug("TabularPyTorchDataset Dataset saved to a file: \n %s" % dataobj_file)

    @classmethod
    def load(cls, file_prefix=""):
        """ Additional naming changes will be appended to end of file_prefix (must contain full absolute path) """
        dataobj_file = file_prefix + cls.DATAOBJ_SUFFIX
        dataset: TabularTorchDataset = torch.load(dataobj_file)
        logger.debug("TabularNN Dataset loaded from a file: \n %s" % dataobj_file)
        return dataset

    def build_loader(self, batch_size, num_workers, is_test=False):
        def worker_init_fn(worker_id):
            np.random.seed(np.random.get_state()[1][0] + worker_id)
        loader = torch.utils.data.DataLoader(self, batch_size=batch_size, num_workers=num_workers,
                                             shuffle=False if is_test else True,
                                             drop_last=False if is_test else True,
                                             worker_init_fn=worker_init_fn)
        return loader
