""" MXNet neural networks for tabular data containing numerical, categorical, and text fields.
    First performs neural network specific pre-processing of the data using sklearn tools.
    Contains separate input modules which are applied to different columns of the data depending on the type of values they contain:
    - Numeric columns are pased through single Dense layer (binary categorical variables are treated as numeric)
    - Categorical columns are passed through separate Embedding layers
    - Text columns are passed through separate LanguageModel layers
    Vectors produced by different input layers are then concatenated and passed to multi-layer MLP model with problem_type determined output layer.
    Hyperparameters are passed as dict params, including options for preprocessing stages.
"""

import contextlib, shutil, tempfile, math, random, warnings, os
from pathlib import Path
from collections import OrderedDict 
import numpy as np
import pandas as pd
import mxnet as mx
from mxnet import nd, autograd, gluon
from mxboard import SummaryWriter
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PowerTransformer, QuantileTransformer, FunctionTransformer

from f3_grail_data_frame_utilities.loaders import load_pkl
from tabular.ml.models.abstract_model import AbstractModel
from tabular.utils.savers import save_pkl
from tabular.ml.constants import BINARY, MULTICLASS, REGRESSION
# TODO these files should be moved eventually:
from tabular.ml.mxnet.categorical_encoders import OneHotMergeRaresHandleUnknownEncoder, OrdinalMergeRaresHandleUnknownEncoder
from tabular.ml.mxnet.tabular_nn_dataset import TabularNNDataset
from tabular.ml.mxnet.embednet import EmbedNet

@contextlib.contextmanager # TODO: keep this?
def make_temp_directory():
    temp_dir = tempfile.mkdtemp()
    try:
        yield temp_dir
    finally:
        shutil.rmtree(temp_dir)

class TabularNeuralNetModel(AbstractModel):
    
    """ Class for neural network models that operate on tabular data. These networks use different types of input layers to process different types of data in various columns.
    
        Attributes:
            types_of_features (dict): keys = 'continuous', 'skewed', 'onehot', 'embed', 'language'; values = column-names of dataframe corresponding to the features of this type
            feature_arraycol_map (OrderedDict): maps feature-name -> list of column-indices in processed_array corresponding to this feature
        self.feature_type_map (OrderedDict): maps feature-name -> feature_type string (options: 'vector', 'embed', 'language')
        processor (sklearn.ColumnTransformer): scikit-learn preprocessor object
        
    """
    
    # Constants used throughout this class:
    # model_internals_file_name = 'model-internals.pkl' # store model internals here
    unique_category_str = '!missing!' # string used to represent missing values and unknown categories for categorical features. Should not appear in the dataset
    metric_map = {REGRESSION: 'MSE', BINARY: 'error_rate', MULTICLASS: 'error_rate'}  # string used to represent different evaluation metrics. metric_map[self.problem_type] produces str corresponding to metric used here.
    model_file_name = 'tabularNN.pkl'
    params_file_name = 'net.params' # Stores parameters of final network
    temp_file_name = 'temp_net.params' # stores temporary network parameters (eg. during the course of training)
    
    def __init__(self, path, name, problem_type, objective_func, features=None, params=None):
        super().__init__(path=path, name=name, model=None, problem_type=problem_type, objective_func=objective_func, features=features)
        """ Create new TabularNeuralNetModel object.
            Args:
                params (dict): various hyperparameters for our neural network model and the NN-specific data processing steps
        """
        #self.bs = params['nn.tabular.bs']
        # self.ps = params['nn.tabular.ps']
        # self.emb_drop = params['nn.tabular.emb_drop']
        # self.lr = params['nn.tabular.lr']
        # self.epochs = params['nn.tabular.epochs']
        # self.metric = params['nn.tabular.metric']
        self.problem_type = problem_type
        self.objective_func = objective_func
        self.feature_types_metadata = None
        self.types_of_features = None
        self.feature_arraycol_map = None
        self.feature_type_map = None
        self.processor = None # data processor
        if params is None:
            self.params = {}
        else:
            self.params = params
        self.set_default_params()
    
    def set_default_params(self):
        """ Specifies hyperparameter values to use by default """
        
        # Configuration-options that we never search over in HPO but user can specify:
        self._use_default_value('num_dataloading_workers', 1) # not searched... depends on num_cpus provided by trial manager
        self._use_default_value('ctx', mx.gpu() if mx.test_utils.list_gpus() else mx.cpu() ) # not searched... depends on num_gpus provided by trial manager
        self._use_default_value('max_epochs', 100)  # maximum number of epochs for training NN
        
        # For data processing:
        self._use_default_value('proc.embed_min_categories', 4)  # apply embedding layer to categorical features with at least this many levels. Features with fewer levels are one-hot encoded. Choose big value to avoid use of Embedding layers
        # Default search space: 3,4,10, 100, 1000
        self._use_default_value('proc.impute_strategy', 'median') # strategy argument of SimpleImputer() used to impute missing numeric values
        # Default search space: ['median', 'mean', 'most_frequent']
        self._use_default_value('proc.max_category_levels', 500) # maximum number of allowed levels per categorical feature
        # Default search space: [10, 100, 200, 300, 400, 500, 1000, 10000]
        self._use_default_value('proc.power_transform_method', 'yeo-johnson') # method argument of PowerTransformer (can alternatively be 'box-cox' but this will fail for features with negative values)
        # Default search space: [10, 100, 200, 300, 400, 500, 1000, 10000]
        self._use_default_value('proc.skew_threshold', 0.5) # numerical features whose absolute skewness is greater than this receive special power-transform preprocessing. Choose big value to avoid using power-transforms
        # Default search space: np.linspace(0.1, 1, 10)
        
        # Hyperparameters for neural net architecture:
        self._use_default_value('layers', None) # List of widths (num_units) for each hidden layer
        # Default search space: List of lists that are manually created
        self._use_default_value('numeric_embed_dim', None) # Size of joint embedding for all numeric+one-hot features.
        # Default search space: TBD
        self._use_default_value('activation', 'relu')
        # Default search space: ['relu', 'elu', 'tanh']
        self._use_default_value('max_layer_width', 2056) # maximum number of hidden units in MLP layer
        # Does not need to be searched by default!
        self._use_default_value('embedding_size_factor', 1.0)
        # Default search space: [0.01 - 100] on log-scale
        self._use_default_value('embed_exponent', 0.56)
         # Does not need to be searched by default!
        self._use_default_value('max_embedding_dim', 500)
        
        # Hyperparameters for neural net training:
        self._use_default_value('batch_size', 64) # batch-size used for NN training
        # Default search space: [16, 32, 64, 128. 256, 512]
        self._use_default_value('loss_function', None) # Loss function used for training
        self._use_default_value('optimizer', 'adam')
        self._use_default_value('learning_rate', 3e-4) # learning rate used for NN training
        self._use_default_value('weight_decay', 1e-8)
        self._use_default_value('clip_gradient', 10.0)
        self._use_default_value('momentum', 0.9) # only for SGD
        self._use_default_value('epochs_wo_improve', max(5, int(self.params['max_epochs']/5.0))) # we terminate training if val accuracy hasn't improved in the last 'epochs_wo_improve' # of epochs
    
    
    def set_net_defaults(self, train_dataset):
        """ Sets dataset-adaptive default values to use for our neural network """
        if self.problem_type == MULTICLASS:
            self.num_classes = train_dataset.num_classes
            self.num_net_outputs = self.num_classes
        elif self.problem_type == REGRESSION:
            self.num_net_outputs = 1
        elif self.problem_type == BINARY:
            self.num_classes = 2
            self.num_net_outputs = 2
        
        if self.params['layers'] is None: # Use default choices for MLP architecture
            if self.problem_type == REGRESSION:
                default_layer_sizes = [256] # overall network will have 3 layers. Input layer, 256-unit hidden layer, output layer.
            elif self.problem_type == BINARY or self.problem_type == MULTICLASS:
                default_sizes = [256] # will be scaled adaptively
                base_size = max(1, min(self.num_net_outputs, 20)/2.0) # scale layer width based on number of classes
                default_layer_sizes = [defaultsize*base_size for defaultsize in default_sizes]
            layer_expansion_factor = np.log10(max(train_dataset.num_examples, 1000)) - 2 # scale layers based on training examples...
            max_layer_width = self.params['max_layer_width']
            self.params['layers'] = [min(max_layer_width, int(layer_expansion_factor*defaultsize)) 
                                     for defaultsize in default_layer_sizes]
        
        if train_dataset.has_vector_features() and self.params['numeric_embed_dim'] is None:
            # Use default choices for numeric embedding size
            vector_dim = train_dataset.dataset._data[train_dataset.vectordata_index].shape[1]  # total dimensionality of vector features
            min_numeric_embed_dim = 16
            max_numeric_embed_dim = self.params['max_layer_width']
            self.params['numeric_embed_dim'] = int(min(max_numeric_embed_dim, max(min_numeric_embed_dim,
                                                    self.params['layers'][0]*np.log10(vector_dim+0.01) )))
        return
    
    def fit(self, X_train, Y_train, X_test=None, Y_test=None):
        """ X_train (pd.DataFrame): training data features (not necessarily preprocessed yet)
            X_test (pd.DataFrame): test data features (should have same column names as Xtrain)
            Y_train (pd.Series): 
            Y_test (pd.Series): are pandas Series 
        """
        if self.feature_types_metadata is None:
            raise ValueError("Trainer class must set feature_types_metadata for this model")
        X_train = self.preprocess(X_train)  # TODO from grail: Handle cases where features have been removed due to tuning, currently crashes.
        print('features: ', self.features)
        train_dataset = self.process_data(X_train, Y_train, is_test=False) # Dataset object
        if X_test is not None:
            X_test = self.preprocess(X_test)
            test_dataset = self.process_data(X_test, Y_test, is_test=True) # Dataset object to use for validation
        else:
            test_dataset = None
        print("Training data has: %d examples, %d features (%d vector, %d embedding, %d language)" % 
              (train_dataset.num_examples, train_dataset.num_features, 
               len(train_dataset.feature_groups['vector']), len(train_dataset.feature_groups['embed']),
               len(train_dataset.feature_groups['language']) ))
        # train_dataset.save()
        # test_dataset.save()
        # self._save_preprocessor() # TODO: should save these things for hyperparam tunning. Need one HP tuner for network-specific HPs, another for preprocessing HPs...
        self.get_net(train_dataset)
        self.train_net(train_dataset, test_dataset, initialize=True, setup_trainer=True)
        """
        # TODO: if we don't want to save intermediate network parameters, need to do something like this to clean them up after training.
        with make_temp_directory() as temp_dir:
            save_callback = SaveModelCallback(self.model, monitor=self.metric, mode=save_callback_mode, name=self.name)
            with progress_disabled_ctx(self.model) as model:
                original_path = model.path
                model.path = Path(temp_dir)
                model.fit_one_cycle(self.epochs, self.lr, callbacks=save_callback)

                # Load the best one and export it
                model.load(self.name)
                print(f'Model validation metrics: {model.validate()}')
                model.path = original_path\
        """
    
    def get_net(self, train_dataset, initialize=True, setup_trainer=True):
        """ Creates a Gluon neural net and context for this dataset.
            Also sets up trainer/optimizer as necessary.
        """
        self.set_net_defaults(train_dataset)
        self.ctx = self.params['ctx']
        net = EmbedNet(train_dataset=train_dataset, params=self.params,
                       num_net_outputs=self.num_net_outputs)
        self.architecture_desc = net.architecture_desc # Description of network architecture
        self.net_filename = self.path + self.temp_file_name
        self.model= net
        return
    
    def train_net(self, train_dataset, test_dataset=None, 
                  initialize=True, setup_trainer=True):
        """ Trains neural net on given train dataset, early stops based on test_dataset """
        if initialize:
            self.model.collect_params().initialize(ctx=self.ctx)
            self.model.hybridize()
        if setup_trainer:
            self.setup_trainer()
        best_val_metric = np.inf # smaller = better (aka Error rate for classification)
        val_metric = None
        best_val_epoch = 0
        best_train_epoch = 0 # epoch with best training loss so far
        best_train_loss = np.inf # smaller = better
        max_epochs = self.params['max_epochs']
        for e in range(max_epochs):
            cumulative_loss = 0
            for batch_idx, data_batch in enumerate(train_dataset.dataloader):
                data_batch = train_dataset.format_batch_data(data_batch, self.ctx)
                with autograd.record():
                    output = self.model(data_batch)
                    labels = data_batch['label']
                    loss = self.loss_func(output, labels)
                    # print(str(nd.mean(loss).asscalar()), end="\r") # prints per-batch losses
                loss.backward()
                self.optimizer.step(labels.shape[0])
                cumulative_loss += nd.sum(loss).asscalar()
            train_loss = cumulative_loss/float(train_dataset.num_examples) # training loss this epoch
            if test_dataset is not None:
                val_metric = self.evaluate_metric(test_dataset) # Evaluate after each epoch
            if test_dataset is None or val_metric <= best_val_metric: # keep training while validation accuracy remains the same.
                best_val_metric = val_metric
                best_val_epoch = e
                self.model.save_parameters(self.net_filename)
            if test_dataset is not None:
                print("Epoch %s.  Train loss: %s, Val %s: %s" %
                  (e, train_loss, self.metric_map[self.problem_type], val_metric))
                self.summary_writer.add_scalar(tag='val_'+self.metric_map[self.problem_type], 
                                               value=val_metric, global_step=e)
            else:
                print("Epoch %s.  Train loss: %s" % (e, train_loss))
            self.summary_writer.add_scalar(tag='train_loss', value=train_loss, global_step=e)
            # TODO: callback / logger here!!
            if e - best_val_epoch > self.params['epochs_wo_improve']:
                break
            if e == 0: # speical actions during first epoch:
                print(self.model)  # TODO: remove?
        self.model.load_parameters(self.net_filename) # Revert back to best model
        if test_dataset is None: # evaluate one final time:
            print("Best model found in epoch %d" % best_val_epoch)
        else:
            final_val_metric = self.evaluate_metric(test_dataset)
            print("Best model found in epoch %d. Val %s: %s" %
                  (best_val_epoch, self.metric_map[self.problem_type], final_val_metric))
        return
    
    def evaluate_metric(self, dataset):
        """ Evaluates metric on the given dataset (TabularNNDataset object). 
            Returns error-rate in the case of classification, MSE for regression.
        """
        if self.problem_type == REGRESSION:
            mx_metric = mx.metric.MSE()
        else:
            mx_metric = mx.metric.Accuracy()
        for batch_idx, data_batch in enumerate(dataset.dataloader):
            data_batch = dataset.format_batch_data(data_batch, self.ctx)
            preds = self.model(data_batch)
            mx_metric.update(preds=preds, labels=data_batch['label']) # argmax not needed, even for classification
        if self.problem_type == REGRESSION:
            return mx_metric.get()[1]
        else:
            return 1.0 - mx_metric.get()[1] # error rate
    
    def predict_proba(self, X, preprocess=True):
        """ To align predict wiht abstract_model API. 
            Preprocess here only refers to feature processing stesp done by all AbstratModel objects, 
            not tabularNN-specific preprocessing steps.
        """
        if preprocess:
            X = self.preprocess(X)
        return self._predict_tabular_data(new_data=X, process=True, predict_proba=True)
    
    def _predict_tabular_data(self, new_data, process=True, predict_proba=True): # TODO ensure API lines up with tabular.Model class.
        """ Specific TabularNN method to produce predictions on new (unprocessed) data. 
            Returns 1D numpy array unless predict_proba=True and task is multi-class classification (not binary).
            Args:
                new_data (pd.Dataframe or TabularNNDataset): new data to make predictions on.
                If you want to make prediction for just a single row of new_data, pass in: new_data.iloc[[row_index]]
                process (bool): should new data be processed (if False, new_data must be TabularNNDataset)
                predict_proba (bool): should we output class-probabilities (not used for regression)
        """
        if process:
            new_data = self.process_data(new_data, labels=None, is_test=True)
        if not isinstance(new_data, TabularNNDataset):
            raise ValueError("new_data must of of type TabularNNDataset if preprocess=False")
        if self.problem_type == REGRESSION or not predict_proba:
            preds = nd.zeros((new_data.num_examples,1))
        else:
            preds = nd.zeros((new_data.num_examples, self.num_net_outputs))
        i = 0
        for batch_idx, data_batch in enumerate(new_data.dataloader):
            data_batch = new_data.format_batch_data(data_batch, self.ctx)
            preds_batch = self.model(data_batch)
            batch_size = len(preds_batch)
            if self.problem_type != REGRESSION: 
                if not predict_proba: # need to take argmax
                    preds_batch = nd.argmax(preds_batch, axis=1, keepdims=True)
                else: # need to take softmax
                    preds_batch = nd.softmax(preds_batch, axis=1)
            preds[i:(i+batch_size)] = preds_batch
            i = i+batch_size
        if self.problem_type == REGRESSION or not predict_proba:
            return preds.asnumpy().flatten() # return 1D numpy array
        elif self.problem_type == BINARY and predict_proba:
            return preds[:,1].asnumpy() # for binary problems, only return P(Y==1)
        return preds.asnumpy() # return 2D numpy array
    
    def process_data(self, df, labels = None, is_test=True):
        """ Process train or test DataFrame into a form fit for neural network models.
        Args:
            df (pd.DataFrame): Data to be processed (X)
            labels (pd.Series): labels to be processed (y)
            test (bool): Is this test data where each datapoint should be processed separately using predetermined preprocessing steps. 
                         Otherwise preprocessor uses all data to determine propreties like best scaling factors, number of categories, etc.
        Returns:
            Dataset object
        """
        if set(df.columns) != set(self.features):
            raise ValueError("Column names in provided Dataframe do not match self.features")
        if labels is not None and len(labels) != len(df):
            raise ValueError("Number of examples in Dataframe does not match number of labels")
        if not is_test:
            return self.process_train_data(df, labels)
        # Otherwise we are processing test data:
        if (self.processor is None or self.types_of_features is None 
           or self.feature_arraycol_map is None or self.feature_type_map is None):
            raise ValueError("Need to process training data before test data")
        processed_array = self.processor.transform(df) # 2D numpy array. self.feature_arraycol_map, self.feature_type_map have been previously set while processing training data.
        return TabularNNDataset(processed_array, self.feature_arraycol_map, self.feature_type_map, self.params, labels=labels, is_test=True,
                                problem_type = self.problem_type)
    
    def process_train_data(self, df, labels):
        """ Preprocess training data and create self.processor object that can be used to process future data.
            This method should only be used once per TabularNeuralNetModel object, otherwise will produce Warning.
             
        # TODO no label processing for now
        # TODO: language features are ignored for now
        # TODO: how to add new features such as time features and remember to do the same for test data?
        # TODO: no filtering of data-frame columns based on statistics, e.g. categorical columns with all unique variables or zero-variance features. 
                This should be done in default_learner class for all models not just TabularNeuralNetModel...
        
        Here is old Grail code for column-filtering of data-frame Xtrain based on statistics:
        try:
            X_train_stats = X_train.describe(include='all').T.reset_index()
            cols_to_drop = X_train_stats[(X_train_stats['unique'] > self.max_unique_categorical_values) | (X_train_stats['unique'].isna())]['index'].values
        except:
            cols_to_drop = []
        cols_to_keep = [col for col in list(X_train.columns) if col not in cols_to_drop]
        cols_to_use = [col for col in self.cat_names if col in cols_to_keep]
        print(f'Using {len(cols_to_use)}/{len(self.cat_names)} categorical features')
        self.cat_names = cols_to_use
        print(f'Using {len(self.cont_names)} cont features')
        """
        if labels is None:
            raise ValueError("Attempting process training data without labels")
        self.types_of_features = self._get_types_of_features(df) # dict with keys: : 'continuous', 'skewed', 'onehot', 'embed', 'language', values = column-names of df
        print("Feature types: ", self.types_of_features)
        self.processor = self._create_preprocessor()
        processed_array = self.processor.fit_transform(df) # 2D numpy array
        self.feature_arraycol_map = self._get_feature_arraycol_map() # OrderedDict of feature-name -> list of column-indices in processed_array corresponding to this feature
        # print(self.feature_arraycol_map)
        self.feature_type_map = self._get_feature_type_map() # OrderedDict of feature-name -> feature_type string (options: 'vector', 'embed', 'language')
        # print(self.feature_type_map)
        return TabularNNDataset(processed_array, self.feature_arraycol_map, self.feature_type_map, self.params, labels=labels, is_test=False)
    
    def setup_trainer(self):
        """ Set up stuff needed for training: 
            optimizer, loss, and summary writer (for mxboard).
            Network must first be initialized before this. 
        """
        self.summary_writer = SummaryWriter(logdir=self.path, flush_secs=10)
        optimizer_opts = {'learning_rate': self.params['learning_rate'],  
            'wd': self.params['weight_decay'], 'clip_gradient': self.params['clip_gradient']}
        if self.params['optimizer'] == 'sgd':
            optimizer_opts['momentum'] = self.params['momentum']
            self.optimizer = gluon.Trainer(self.model.collect_params(), 'sgd', optimizer_opts)
        elif self.params['optimizer'] == 'adam':
            self.optimizer = gluon.Trainer(self.model.collect_params(), 'adam', optimizer_opts)
        else:
            raise ValueError("Unknown optimizer specified: %s" % self.params['optimizer'])
        if self.params['loss_function'] is None:
            if self.problem_type == REGRESSION:
                self.params['loss_function'] = gluon.loss.L1Loss()
            else:
                self.params['loss_function'] = gluon.loss.SoftmaxCrossEntropyLoss(from_logits=self.model.from_logits)
        self.loss_func = self.params['loss_function']
        return
    
    # Helper functions for tabular NN:
    
    def __get_feature_type_if_present(self, feature_type):
        """ Returns crude categorization of feature types """
        return self.feature_types_metadata[feature_type] if feature_type in self.feature_types_metadata else []
    
    def _get_types_of_features(self, df):
        """ Returns dict with keys: : 'continuous', 'skewed', 'onehot', 'embed', 'language', values = ordered list of feature-names falling into each category.
            Each value is a list of feature-names corresponding to columns in original dataframe.
            TODO: ensure features with zero variance have already been removed before this function is called.
        """
        if self.types_of_features is not None:
            Warning("Attempting to _get_types_of_features for TabularNeuralNetModel, but previously already did this.")
        categorical_featnames = self.__get_feature_type_if_present('object') + self.__get_feature_type_if_present('bool')
        continuous_featnames = self.__get_feature_type_if_present('float') + self.__get_feature_type_if_present('int')
        print("categorical_featnames:", categorical_featnames)
        print("continuous_featnames:", continuous_featnames)
        language_featnames = [] # TODO: not implemented. This should fetch text features present in the data
        if len(categorical_featnames) + len(continuous_featnames) + len(language_featnames) != df.shape[1]:
            raise ValueError("unknown feature types present in DataFrame")
        
        types_of_features = {'continuous': [], 'skewed': [], 'onehot': [], 'embed': [], 'language': []}
        # continuous = numeric features to rescale
        # skewed = features to which we will apply power (ie. log / box-cox) transform before normalization
        # onehot = features to one-hot encode (unknown categories for these features encountered at test-time are encoded as all zeros). We one-hot encode any features encountered that only have two unique values.
        for feature in self.features:
            feature_data = df[feature] # pd.Series
            num_unique_vals = len(feature_data.unique())
            if num_unique_vals == 2: # will be onehot encoded
                types_of_features['onehot'].append(feature)
            elif feature in continuous_featnames:
                if np.abs(feature_data.skew()) > self.params['proc.skew_threshold']:
                    types_of_features['skewed'].append(feature)
                else:
                    types_of_features['continuous'].append(feature)
            elif feature in categorical_featnames:
                if num_unique_vals >= self.params['proc.embed_min_categories']: # sufficiently many cateories to warrant learned embedding dedicated to this feature
                    types_of_features['embed'].append(feature)
                else:
                    types_of_features['onehot'].append(feature)
            elif feature in language_featnames:
                types_of_features['language'].append(feature)
        return types_of_features
    
    def _get_feature_arraycol_map(self):
        """ Returns OrderedDict of feature-name -> list of column-indices in processed data array corresponding to this feature """
        feature_preserving_transforms = set(['continuous','skewed', 'ordinal', 'language']) # these transforms do not alter dimensionality of feature
        feature_arraycol_map = {} # unordered version
        current_colindex = 0
        for transformer in self.processor.transformers_:
            transformer_name = transformer[0]
            transformed_features = transformer[2]
            if transformer_name in feature_preserving_transforms:
                for feature in transformed_features:
                    if feature in feature_arraycol_map:
                        raise ValueError("same feature is processed by two different column transformers: %s" % feature)
                    feature_arraycol_map[feature] = [current_colindex]
                    current_colindex += 1
            elif transformer_name == 'onehot':
                oh_encoder = [step for (name, step) in transformer[1].steps if name == 'onehot'][0]
                for i in range(len(transformed_features)):
                    feature = transformed_features[i]
                    if feature in feature_arraycol_map:
                        raise ValueError("same feature is processed by two different column transformers: %s" % feature)
                    oh_dimensionality = len(oh_encoder.categories_[i])
                    feature_arraycol_map[feature] = list(range(current_colindex, current_colindex+oh_dimensionality))
                    current_colindex += oh_dimensionality
            else:
                raise ValueError("unknown transformer encountered: %s" % transformer_name)
        if set(feature_arraycol_map.keys()) != set(self.features):
            raise ValueError("failed to account for all features when determining column indices in processed array") 
        return OrderedDict([(key, feature_arraycol_map[key]) for key in feature_arraycol_map])
    
    def _get_feature_type_map(self):
        """ Returns OrderedDict of feature-name -> feature_type string (options: 'vector', 'embed', 'language') """
        if self.feature_arraycol_map is None:
            raise ValueError("must first call _get_feature_arraycol_map() before _get_feature_type_map()")
        vector_features = self.types_of_features['continuous'] + self.types_of_features['skewed'] + self.types_of_features['onehot']
        feature_type_map = OrderedDict()
        for feature_name in self.feature_arraycol_map:
            if feature_name in vector_features:
                feature_type_map[feature_name] = 'vector'
            elif feature_name in self.types_of_features['embed']:
                feature_type_map[feature_name] = 'embed'
            elif feature_name in self.types_of_features['language']:
                feature_type_map[feature_name] = 'language'
            else: 
                raise ValueError("unknown feature type encountered")
        return feature_type_map
    
    def _create_preprocessor(self):
        """ Defines data encoders used to preprocess different data types and creates instance variable which is sklearn ColumnTransformer object """
        if self.processor is not None:
            Warning("Attempting to process training data for TabularNeuralNetModel, but previously already did this.")
        continuous_features = self.types_of_features['continuous']
        skewed_features = self.types_of_features['skewed']
        onehot_features = self.types_of_features['onehot']
        embed_features = self.types_of_features['embed']
        language_features = self.types_of_features['language']
        transformers = [] # order of various column transformers in this list is important!
        if len(continuous_features) > 0:
            continuous_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy=self.params['proc.impute_strategy'])),
                ('scaler', StandardScaler())])
            transformers.append( ('continuous', continuous_transformer, continuous_features) )
        if len(skewed_features) > 0:
            power_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy=self.params['proc.impute_strategy'])),
                ('power', PowerTransformer(method=self.params['proc.power_transform_method'])) ])
            transformers.append( ('skewed', power_transformer, skewed_features) )
        if len(onehot_features) > 0:
            onehot_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='constant', fill_value=self.unique_category_str)),
                ('onehot', OneHotMergeRaresHandleUnknownEncoder(max_levels=self.params['proc.max_category_levels'],sparse=False)) ]) # test-time unknown values will be encoded as all zeros vector
            transformers.append( ('onehot', onehot_transformer, onehot_features) )
        if len(embed_features) > 0: # Ordinal transformer applied to convert to-be-embedded categorical features to integer levels
            ordinal_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='constant', fill_value=self.unique_category_str)),
                ('ordinal', OrdinalMergeRaresHandleUnknownEncoder(max_levels=self.params['proc.max_category_levels'])) ]) # returns 0-n when max_category_levels = n-1. category n is reserved for unknown test-time categories.
            transformers.append( ('ordinal', ordinal_transformer, embed_features) )
        if len(language_features) > 0:
            raise NotImplementedError("language_features cannot be used at the moment")
        return ColumnTransformer(transformers=transformers) # numeric features are processed in the same order as in numeric_features vector, so feature-names remain the same.
    
    def save(self):
        if self.model is not None:
            self.model.save_parameters(self.path + self.params_file_name)
        temp_model = self.model
        temp_sw = self.summary_writer
        self.model = None
        self.summary_writer = None
        save_pkl.save(path=self.path + self.model_file_name, object=self)
        self.model = temp_model
        self.summary_writer = temp_sw
    
    @classmethod
    def load(cls, path, reset_paths=False):
        load_path = path + cls.model_file_name
        if not reset_paths:
            obj = load_pkl.load(path=load_path)
        else:
            obj = load_pkl.load(path=load_path)
            obj.set_contexts(path)
        obj.model = EmbedNet(architecture_desc=obj.architecture_desc) # recreate network from architecture description
        # TODO: maybe need to initialize/hybridize??
        obj.model.load_parameters(path + cls.params_file_name)
        obj.summary_writer = SummaryWriter(logdir=obj.path, flush_secs=10)
        return obj
    
    def _use_default_value(self, param_name, param_value):
        if param_name not in self.params:
            self.params[param_name] = param_value
    


"""  General TODOs:

- OrdinalEncoder class in sklearn currently cannot handle rare categories or unknown ones at test-time, so we have created our own Encoder in category_encoders.py
There is open PR in sklearn to address this: https://github.com/scikit-learn/scikit-learn/pull/13833/files
Currently, our code uses category_encoders package (BSD license) instead: https://github.com/scikit-learn-contrib/categorical-encoding
Once PR is merged into sklearn, may want to switch: category_encoders.Ordinal -> sklearn.preprocessing.OrdinalEncoder in preprocess_train_data()


TODO: how to save preprocessed data so that we can do HPO of neural net hyperparameters more efficiently, while also doing HPO of preprocessing hyperparameters?
      Naive full HPO method requires redoing preprocessing in each trial even if we did not change preprocessing hyperparameters.
      Alternative is we save each proprocessed dataset & corresponding TabularNeuralNetModel object with its unique param names in the file. Then when we try a new HP-config, we first try loading from file if one exists.

TODO: feature_types_metadata must be set outside of model class in trainer.

TODO: enable seeds?

TODO: test regression + multiclassification

TODO: benchmark against fastAI

TODO: issue: embedding layers learn much slower than Dense layers. Need to carefully initialize

"""
