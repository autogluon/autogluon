import logging

import torch
import torch.nn as nn
import numpy as np
import pandas as pd

from .embednet import getEmbedSizes

logger = logging.getLogger(__name__)


class NeuralMultiQuantileRegressor(nn.Module):
    def __init__(self,
                 quantile_levels,
                 train_dataset=None,
                 params=None,
                 architecture_desc=None,
                 device=None):
        if (architecture_desc is None) and (train_dataset is None or params is None):
            raise ValueError("train_dataset, params cannot = None if architecture_desc=None")
        super(NeuralMultiQuantileRegressor, self).__init__()
        self.register_buffer('quantile_levels', torch.Tensor(quantile_levels).float().reshape(1, -1))
        self.device = torch.device('cpu') if device is None else device
        if architecture_desc is None:
            # adpatively specify network architecture based on training dataset
            self.from_logits = False
            self.has_vector_features = train_dataset.has_vector_features()
            self.has_embed_features = train_dataset.num_embed_features() > 0
            self.has_language_features = train_dataset.num_language_features() > 0
            if self.has_embed_features:
                num_categs_per_feature = train_dataset.getNumCategoriesEmbeddings()
                embed_dims = getEmbedSizes(train_dataset, params, num_categs_per_feature)
            if self.has_vector_features:
                vector_dims = train_dataset.data_list[train_dataset.vectordata_index].shape[-1]
        else:
            # ignore train_dataset, params, etc. Recreate architecture based on description:
            self.architecture_desc = architecture_desc
            self.has_vector_features = architecture_desc['has_vector_features']
            self.has_embed_features = architecture_desc['has_embed_features']
            self.has_language_features = architecture_desc['has_language_features']
            self.from_logits = architecture_desc['from_logits']
            params = architecture_desc['params']
            if self.has_embed_features:
                num_categs_per_feature = architecture_desc['num_categs_per_feature']
                embed_dims = architecture_desc['embed_dims']
            if self.has_vector_features:
                vector_dims = architecture_desc['vector_dims']
        # init input / output size
        input_size = 0
        output_size = self.quantile_levels.size()[-1]

        # define embedding layer:
        if self.has_embed_features:
            self.embed_blocks = nn.ModuleList()
            for i in range(len(num_categs_per_feature)):
                self.embed_blocks.append(nn.Embedding(num_embeddings=num_categs_per_feature[i],
                                                      embedding_dim=embed_dims[i]))
                input_size += embed_dims[i]

        # language block not supported
        if self.has_language_features:
            self.text_block = None
            raise NotImplementedError("text data cannot be handled")

        # update input size
        if self.has_vector_features:
            input_size += vector_dims

        # activation
        act_fn = nn.Identity()
        if params['activation'] == 'elu':
            act_fn = nn.ELU()
        elif params['activation'] == 'relu':
            act_fn = nn.ReLU()
        elif params['activation'] == 'tanh':
            act_fn = nn.Tanh()

        # layers
        layers = [nn.Linear(input_size, params['hidden_size']), act_fn]
        for _ in range(params['num_layers'] - 1):
            layers.append(nn.Dropout(params['dropout_prob']))
            layers.append(nn.Linear(params['hidden_size'], params['hidden_size']))
            layers.append(act_fn)
        layers.append(nn.Linear(params['hidden_size'], output_size))
        self.main_block = nn.Sequential(*layers)

        # set range for output
        y_range = params['y_range']  # Used specifically for regression. = None for classification.
        self.y_constraint = None  # determines if Y-predictions should be constrained
        if y_range is not None:
            if y_range[0] == -np.inf and y_range[1] == np.inf:
                self.y_constraint = None  # do not worry about Y-range in this case
            elif y_range[0] >= 0 and y_range[1] == np.inf:
                self.y_constraint = 'nonnegative'
            elif y_range[0] == -np.inf and y_range[1] <= 0:
                self.y_constraint = 'nonpositive'
            else:
                self.y_constraint = 'bounded'
            self.y_lower = y_range[0]
            self.y_upper = y_range[1]
            self.y_span = self.y_upper - self.y_lower

        # for huber loss
        self.alpha = params['alpha']

        if architecture_desc is None:  # Save Architecture description
            self.architecture_desc = {'has_vector_features': self.has_vector_features,
                                      'has_embed_features': self.has_embed_features,
                                      'has_language_features': self.has_language_features,
                                      'params': params,
                                      'from_logits': self.from_logits}
            if self.has_embed_features:
                self.architecture_desc['num_categs_per_feature'] = num_categs_per_feature
                self.architecture_desc['embed_dims'] = embed_dims
            if self.has_vector_features:
                self.architecture_desc['vector_dims'] = vector_dims

    def init_params(self):
        for layer in self.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def forward(self, data_batch):
        input_data = []
        if self.has_vector_features:
            input_data.append(data_batch['vector'].to(self.device))
        if self.has_embed_features:
            embed_data = data_batch['embed']
            for i in range(len(self.embed_blocks)):
                input_data.append(self.embed_blocks[i](embed_data[i].to(self.device)))
        if self.has_language_features:
            pass

        if len(input_data) > 1:
            input_data = torch.cat(input_data, dim=1)
        else:
            input_data = input_data[0]

        output_data = self.main_block(input_data)

        # output with y-range
        if self.y_constraint is None:
            return output_data
        else:
            if self.y_constraint == 'nonnegative':
                return self.y_lower + torch.abs(output_data)
            elif self.y_constraint == 'nonpositive':
                return self.y_upper - torch.abs(output_data)
            else:
                return torch.sigmoid(output_data) * self.y_span + self.y_lower

    def monotonize(self, input_data):
        # number of quantiles
        num_quantiles = input_data.size()[-1]

        # split into below 50% and above 50%
        idx_50 = num_quantiles // 2

        # if a small number of quantiles are estimated or quantile levels are not centered at 0.5
        if num_quantiles < 3 or self.quantile_levels[0, idx_50] != 0.5:
            return input_data

        # below 50%
        below_50 = input_data[:, :(idx_50 + 1)].contiguous()
        below_50 = torch.flip(torch.cummin(torch.flip(below_50, [-1]), -1)[0], [-1])

        # above 50%
        above_50 = input_data[:, idx_50:].contiguous()
        above_50 = torch.cummax(above_50, -1)[0]

        # refined output
        ordered_data = torch.cat([below_50[:, :-1], above_50], -1)
        return ordered_data

    def huber_pinball_loss(self, input_data, target_data):
        error_data = target_data.contiguous().reshape(-1, 1) - input_data
        loss_data = torch.where(torch.abs(error_data) < self.alpha,
                                0.5 * error_data * error_data,
                                self.alpha * (torch.abs(error_data) - 0.5 * self.alpha))
        loss_data /= self.alpha

        scale = torch.where(error_data >= 0,
                            torch.ones_like(error_data) * self.quantile_levels,
                            torch.ones_like(error_data) * (1 - self.quantile_levels))
        loss_data *= scale
        return loss_data.mean()

    def margin_loss(self, input_data, margin_scale=0.0001):
        # number of samples
        batch_size, num_quantiles = input_data.size()

        # compute margin loss (batch_size x output_size(above) x output_size(below))
        error_data = input_data.unsqueeze(1) - input_data.unsqueeze(2)

        # margin data (num_quantiles x num_quantiles)
        margin_data = self.quantile_levels.permute(1, 0) - self.quantile_levels
        margin_data = torch.tril(margin_data, -1) * margin_scale

        # compute accumulated margin
        loss_data = torch.tril(error_data + margin_data, diagonal=-1)
        loss_data = loss_data.relu()
        loss_data = loss_data.sum() / float(batch_size * (num_quantiles * num_quantiles - num_quantiles) * 0.5)
        return loss_data

    def compute_loss(self,
                     data_batch,
                     weight=0.0,
                     margin=0.0):
        # train mode
        self.train()
        predict_data = self(data_batch)
        target_data = data_batch['label'].to(self.device)

        # get prediction and margin loss
        if margin > 0.0:
            m_loss = self.margin_loss(predict_data)
        else:
            m_loss = 0.0

        h_loss = (1 - weight) * self.huber_pinball_loss(self.monotonize(predict_data), target_data).mean()
        h_loss += weight * self.huber_pinball_loss(predict_data, target_data).mean()
        return h_loss + margin * m_loss

    def predict(self, input_data):
        self.eval()
        with torch.no_grad():
            predict_data = self(input_data)
            predict_data = self.monotonize(predict_data)
            return predict_data.data.cpu().numpy()


class TabularPyTorchDataset(torch.utils.data.Dataset):
    """
        This class is following the structure of TabularNNDataset in tabular_nn_dataset.py

        Class for preprocessing & storing/feeding data batches used by tabular data quantile (pytorch) neural networks.
        Assumes entire dataset can be loaded into numpy arrays.
        Original Data table may contain numerical, categorical, and text (language) fields.

        Attributes:
            data_list (list[np.array]): Contains the raw data. Different indices in this list correspond to different
                                        types of inputs to the neural network (each is 2D array). All vector-valued
                                        (continuous & one-hot) features are concatenated together into a single index
                                        of the dataset.
            data_desc (list[str]): Describes the data type of each index of dataset
                                   (options: 'vector','embed_<featname>', 'language_<featname>')
            embed_indices (list): which columns in dataset correspond to embed features (order matters!)
            language_indices (list): which columns in dataset correspond to language features (order matters!)
            vecfeature_col_map (dict): maps vector_feature_name ->  columns of dataset._data[vector] array that
                                       contain the data for this feature
            feature_dataindex_map (dict): maps feature_name -> i such that dataset._data[i] = data array for
                                          this feature. Cannot be used for vector-valued features,
                                          instead use vecfeature_col_map
            feature_groups (dict): maps feature_type (ie. 'vector' or 'embed' or 'language') to list of feature
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
    DATAOBJ_SUFFIX = '_tabdataset_pytorch.pt'

    def __init__(self,
                 processed_array,
                 feature_arraycol_map,
                 feature_type_map,
                 labels=None):
        """ Args:
            processed_array: 2D numpy array returned by preprocessor. Contains raw data of all features as columns
            feature_arraycol_map (OrderedDict): Mapsfeature-name -> list of column-indices in processed_array
                                                corresponding to this feature
            feature_type_map (OrderedDict): Maps feature-name -> feature_type string
                                            (options: 'vector', 'embed', 'language')
            labels (pd.Series): list of labels (y) if available
        """
        self.num_examples = processed_array.shape[0]
        self.num_features = len(feature_arraycol_map)
        if feature_arraycol_map.keys() != feature_type_map.keys():
            raise ValueError("feature_arraycol_map and feature_type_map must share same keys")
        self.feature_groups = {'vector': [], 'embed': [], 'language': []}
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
        if labels is not None and len(labels) != self.num_examples:
            raise ValueError("number of labels and training examples do not match")

        self.data_desc = []
        self.data_list = []
        self.label_index = None
        self.vectordata_index = None
        self.vecfeature_col_map = {}
        self.feature_dataindex_map = {}

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

        # language data
        if len(self.feature_groups['language']) > 0:
            for feature in feature_type_map:
                if feature_type_map[feature] == 'language':
                    feature_colinds = feature_arraycol_map[feature]
                    self.data_list.append(processed_array[:, feature_colinds].atype('int64').flatten())
                    self.data_desc.append('language')
                    self.feature_dataindex_map[feature] = len(self.data_list) - 1

        # output (target) data
        if labels is not None:
            labels = np.array(labels)
            self.data_desc.append("label")
            self.label_index = len(self.data_list)
            self.data_list.append(labels.astype('float32').reshape(-1, 1))
        self.embed_indices = [i for i in range(len(self.data_desc)) if 'embed' in self.data_desc[i]]
        self.language_indices = [i for i in range(len(self.data_desc)) if 'language' in self.data_desc[i]]

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
        if self.num_language_features() > 0:
            output_dict['language'] = []
            for i in self.language_indices:
                output_dict['language'].append(self.data_list[i][idx])
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

    def num_language_features(self):
        """ Returns number of language features in this dataset """
        return len(self.feature_groups['language'])

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
        nonvector_featuretypes = set(['embed', 'language'])
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
        torch.save(self, dataobj_file)
        logger.debug("TabularPyTorchDataset Dataset saved to a file: \n %s" % dataobj_file)

    @classmethod
    def load(cls, file_prefix=""):
        """ Additional naming changes will be appended to end of file_prefix (must contain full absolute path) """
        dataobj_file = file_prefix + cls.DATAOBJ_SUFFIX
        dataset: TabularPyTorchDataset = torch.load(dataobj_file)
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
