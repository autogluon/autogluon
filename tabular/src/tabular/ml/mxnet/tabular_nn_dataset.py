import contextlib, shutil, tempfile, math, random, warnings
from pathlib import Path
from collections import OrderedDict
import numpy as np
import pandas as pd
import mxnet as mx
from mxnet import nd, autograd, gluon


class TabularNNDataset:
    """ Class for preprocessing & storing/feeding data batches used by tabular data neural networks. Assumes entire dataset can be loaded into numpy arrays.
        Original Data table may contain numerical, categorical, and text (language) fields.
        
        Attributes:
            dataset (mxnet.gluon.data.dataset): Contains the raw data (use dataset._data to access). 
                                                Different indices in this list correspond to different types of inputs to the neural network (each is 2D array)
                                                All vector-valued (continuous & one-hot) features are concatenated together into a single index of the dataset.
            data_desc (list[str]): Describes the data type of each index of dataset (options: 'vector','embed_<featname>', 'language_<featname>')
            dataloader (mxnet.gluon.data.DataLoader): Loads batches of data from dataset for neural net training and inference.
            
            vecfeature_col_map (dict): maps vector_feature_name ->  columns of dataset._data[vector] array that contain the data for this feature
            feature_dataindex_map (dict): maps feature_name -> i such that dataset._data[i] = data array for this feature. Cannot be used for vector-valued features, instead use vecfeature_col_map
            feature_groups (dict): maps feature_type (ie. 'vector' or 'embed' or 'language') to list of feature names of this type (empty list if there are no features of this type)
            vectordata_index (int): describes which element of the dataset._data list holds the vector data matrix (access via self.dataset._data[self.vectordata_index]); None if no vector features
            label_index (int): describing which element of the dataset._data list holds labels (access via self.dataset._data[self.label_index]); None if no labels
            num_examples (int): number of examples in this dataset
            num_features (int): number of features (we only consider original variables as features, so num_features may not correspond to dimensionality of the data eg in the case of one-hot encoding)
    """
    
    def __init__(self, processed_array, feature_arraycol_map, feature_type_map, params, labels=None, is_test=True):
        """ Args:
                processed_array: 2D numpy array returned by preprocessor. Contains raw data of all features as columns
                feature_arraycol_map (OrderedDict): Mapsfeature-name -> list of column-indices in processed_array corresponding to this feature
                feature_type_map (OrderedDict): Maps feature-name -> feature_type string (options: 'vector', 'embed', 'language')
                params (dict): various hyperparameters for our neural network model and the NN-specific data processing steps
        """
        self.params = params
        self.is_test = is_test
        self.num_examples = processed_array.shape[0]
        self.num_features = len(feature_arraycol_map) # number of features (!=dim(processed_array) because some features may be vector-valued, eg one-hot)
        self.batch_size = min(self.num_examples, params['batch_size'])
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
        
        data_list = [] # stores all data of each feature-type in list used to construct MXNet dataset
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
                    self.vecfeature_col_map[feature] = list(range(current_ind, new_last_ind))
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
        
        if labels is not None:
            data_list.append(np.array(labels))
            self.data_desc.append("label")
            self.label_index = len(data_list) - 1
        
        self.dataset = mx.gluon.data.dataset.ArrayDataset(*data_list) # Access ith embedding-feature via: self.dataset._data[self.data_desc.index('embed_'+str(i))].asnumpy()
        self.dataloader = mx.gluon.data.DataLoader(self.dataset, self.batch_size, shuffle= not is_test, 
                                last_batch = 'keep' if is_test else 'rollover',
                                num_workers=self.params['num_dataloading_workers']) # no need to shuffle test data
        
    
    """
    # OLD!
    def __init__(self, params, vector_data=[], vector_features=[], embed_data=[], embed_features=[], embed_numcategories=[], language_data=[],
                 language_features=[], language_processor=None, labels=None, is_test=True):
        "" Args:
                df (DataFrame): contains raw data of all features as columns
                vector_data (list): each element of this list is a 2D numpy array corresponding to data from single vector-valued feature (ie. numeric or one-hot data).
                vector_features (list): feature names for each element of vector_data
                embed_data (list): each element of this list is 2D numpy array of ints representing levels of one categorical feature which should be passed to a Embedding layer
                embed_features (list): feature names for each element of embed_data
                embed_numcategories (list): number of possible categories for each feature passed to an Embedding layer (determines size of Embedding layer)
                language_data (list): each element of this list is NLP-formatted array representing one text field which should be passed to a NLP layer
                language_features (list): feature names for each element of language_data
                language_processor (obj): object for processing raw text data
                labels (1D numpy array): labels for each example (required when is_test = False) 
        ""
        self.is_test = is_test # TODO needed?
        self.params = params
        self.num_examples, self.num_features = df.shape # TODO: where to get these for verification?
        self.batch_size = min(self.num_examples, params['batch_size'])
        self.embed_numcategories = embed_numcategories
        self.language_processor = language_processor
        self.feature_groups = {'vector': vector_features, 'embed': embed_features, 'language': language_features}
        self.vector_feature_inds = {} # vector_feature_inds['featname'] returns the column-indices of self.dataset._data[vector] that correspond to featname (may be multiple columns eg for one-hot feature)
        # Not needed for embed/language features because ith index of data list 
        self.feature_names = vector_features + embed_features + language_features # features must be added to data-list in this order
        
        if len(vector_data) + len(embed_data) + len(language_data) != self.num_features:
            raise ValueError("Dataset() requires data from at least one type of feature")
        if not self.is_test and not labels:
            raise ValueError("labels must be provided when is_test = False")
        if labels is not None and len(labels) != self.num_examples:
            raise ValueError("number of labels and training examples do not match")
        if (len(vector_data) != len(vector_features) or len(embed_data) != len(embed_features) or len(embed_data) != len(embed_numcategories)
            or len(language_data) != len(language_features)):
            raise ValueError("feature names and data-columns must be of equal length for each feature type")
        
        data_list = [] # stores all data of each feature-type in list used to construct MXNet dataset
        self.data_desc = [] # describes feature-type of each index of data_list
        if len(vector_data) > 0:
            vector_datamatrix = np.hstack([vector_data[i] for i in range(len(vector_data))])
            data_list.append(mx.nd.array(vector_datamatrix, dtype='float32')) # Matrix of data from all vector features
            self.data_desc.append("vector")
        for feat_index in range(len(embed_data)):
            data_list.append(embed_data[feat_index]) # may need to convert to dtype= int32
            self.data_desc.append("embed_"+embed_features[feat_index])
        for feat_index in range(len(language_data)):
            data_list.append(language_data[feat_index])
            self.data_desc.append("language_"+)
        if labels is not None :
            data_list.append(labels)
            self.data_desc.append("label")       
        self.dataset = mx.gluon.data.dataset.ArrayDataset(*data_list) # Access ith embedding-feature via: self.dataset._data[self.data_desc.index('embed_'+str(i))].asnumpy()
        self.dataloader = mx.gluon.data.DataLoader(self.dataset, self.batch_size, shuffle= not self.is_test, num_workers=self.params['num_dataloading_workers'])            
    """
    
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
    
    def format_batch_data(self, feature, data_batch):
        """ Returns part of this batch corresponding to data from a single feature 
            Args:
                data_batch (nd.array): the batch of data as provided by self.dataloader 
            Returns:
                formatted_batch (dict): {'vector': array of vector_datamatrix, 'embed': list of embedding features' batch data, 'language': list of language features batch data, 'label': array of labels}
                                         where each key in dict may be missing.
        """
        formatted_batch = {}
        if self.vectordata_index: # None if there is no vector data
            formatted_batch['vector'] = data_batch[self.vectordata_index]
        if self.label_index: # None if there are no labels
            formatted_batch['label'] = data_batch[self.label_index]
        if len(self.feature_groups['embed']) > 0:
            formatted_batch['embed'] = []
            for i in range(len(self.feature_groups['embed'])):
                feature_i = self.feature_groups['embed'][i]
                formatted_batch['embed'].append(self.get_feature_batch(feature_i, data_batch))
        if len(self.feature_groups['language']) > 0:
            formatted_batch['language'] = []
            for i in range(len(self.feature_groups['language'])):
                feature_i = self.feature_groups['language'][i]
                formatted_batch['language'].append(self.get_feature_batch(feature_i, data_batch))
        return formatted_batch
    
    def mask_features_batch(self, features, mask_value, data_batch):
        """ Returns new batch where all values of the indicated features have been replaced by the provided mask_value. 
            Args:
                features (list[str]): list of feature names that should be masked.
                mask_value (float): value of mask which original feature values should be replaced by
                data_batch (nd.array): the batch of data as provided by self.dataloader
            Returns:
                new_batch (nd.array): batch of masked data in same format as data_batch
        """
        return None # TODO
        
     
    def load(file_path):
        return None # TODO save Dataset for reuse during hyperparameter search.
        
    def save(self, file_path):
        return None

