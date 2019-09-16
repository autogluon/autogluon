""" MXNet neural networks for tabular data containing numerical, categorical, and text fields.
    First performs neural network specific pre-processing of the data using sklearn tools.
    Contains separate input modules which are applied to different columns of the data depending on the type of values they contain:
    - Numeric columns are pased through single Dense layer (binary categorical variables are treated as numeric)
    - Categorical columns are passed through separate Embedding layers
    - Text columns are passed through separate LanguageModel layers
    Vectors produced by different input layers are then concatenated and passed to multi-layer MLP model with problem_type determined output layer.
    Hyperparameters are passed as dict params, including options for preprocessing stages.
"""

import contextlib, shutil, tempfile, math, random, warnings
from pathlib import Path
from collections import OrderedDict 
import numpy as np
import pandas as pd
import mxnet as mx
from mxnet import nd, autograd, gluon

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PowerTransformer, QuantileTransformer, FunctionTransformer
# from category_encoders.ordinal import OrdinalEncoder 


from f3_grail_data_frame_utilities.loaders import load_pkl
from tabular.ml.models.abstract_model import AbstractModel
from tabular.utils.savers import save_pkl
from tabular.ml.constants import BINARY, MULTICLASS, REGRESSION


from tabular.ml.mxnet.categorical_encoders import OneHotMergeRaresHandleUnknownEncoder, OrdinalMergeRaresHandleUnknownEncoder # TODO should this file be moved
from tabular.ml.mxnet.tabular_nn_dataset import TabularNNDataset # TODO should this file be moved



@contextlib.contextmanager
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
    model_internals_file_name = 'model-internals.pkl' # store model internals here
    unique_category_str = '!missing!' # string used to represent missing values and unknown categories for categorical features. Should not appear in the dataset
    loss_map = {BINARY: mx.gluon.loss.SoftmaxCrossEntropyLoss(), MULTICLASS: mx.gluon.loss.SoftmaxCrossEntropyLoss(), REGRESSION: mx.gluon.loss.L1Loss()}
    
    def __init__(self, path, name, params, problem_type, objective_func, features=None):
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
        
        # Hyperparameters for neural net:
        self._use_default_value('max_epochs', 100) # maximum number of epochs for training NN
        self._use_default_value('batch_size', 64) # batch-size used for NN training
        self._use_default_value('training_loss', self.loss_map[self.problem_type])  # Loss function used for training
        # Default search space: [16, 32, 64, 128. 256, 512, 1024]
        # self.use_default_value('learning_rate', 1e-3) # learning rate used for NN training
        # Default search space: ...
        
        # Configuration-options that we do NOT search over in HPO:
        self._use_default_value('num_dataloading_workers', 1) # not searched... depends on num_cpus provided by trial manager
        self._use_default_value('ctx', mx.gpu() if mx.test_utils.list_gpus() else mx.cpu() ) # not searched... depends on num_gpus provided by trial manager
    
    def fit(self, X_train, Y_train, X_test=None, Y_test=None):
        """ X_train (pd.DataFrame): training data features (not necessarily preprocessed yet)
            X_test (pd.DataFrame): test data features (should have same column names as Xtrain)
            Y_train (pd.Series): 
            Y_test (pd.Series): are pandas Series. 
        """
        if self.feature_types_metadata is None:
            raise ValueError("Trainer class must set feature_types_metadata for this model")
        X_train = self.preprocess(X_train)  # TODO from grail: Handle cases where features have been removed due to tuning, currently crashes.
        print('features: ', self.features)
        if X_test is not None:
            X_test = self.preprocess(X_test)
        train_dataset = self.process_data(X_train, Y_train, is_test=False) # Dataset object
        test_dataset = self.process_data(X_test, Y_test, is_test=True) # Dataset object to use for validation
        # train_dataset.save()
        # test_dataset.save()
        # self._save_preprocessor() # TODO: save these things for hyperparam tunning
        """
        self.model = self.getNet(train_dataset)
        self.evaluation_metric = metrics_map[self.metric]
        self.trainNet(train_dataset, test_dataset)

        save_callback_mode = 'min' if self.metric == 'mean_absolute_error' else 'auto'
        
        with make_temp_directory() as temp_dir:
            save_callback = SaveModelCallback(self.model, monitor=self.metric, mode=save_callback_mode, name=self.name)
            with progress_disabled_ctx(self.model) as model:
                original_path = model.path
                model.path = Path(temp_dir)
                model.fit_one_cycle(self.epochs, self.lr, callbacks=save_callback)

                # Load the best one and export it
                model.load(self.name)
                print(f'Model validation metrics: {model.validate()}')
                model.path = original_path
        """
        return train_dataset
    
    def predict(self, X, preprocess=True):
        return super().predict(X, preprocess)
    
    def getNet(self):
        """ Instantiates a Hybridized neural network """
        if self.problem_type == REGRESSION or self.problem_type == BINARY:
            layers = [200, 100]
        else:
            base_size = max(len(data.classes) * 2, 100)
            layers = [base_size * 2, base_size]
        model = tabular_learner(data, layers=layers, ps=self.ps, emb_drop=self.emb_drop, metrics=nn_metric)
        print(model)
        return model
    
    def trainNet(self, train_dataset, test_dataset):
        return None
    
    def predict_proba(self, X, preprocess=True): # TODO!
        self.model.data.add_test(TabularList.from_df(X, cat_names=self.cat_names, cont_names=self.cont_names, procs=self.procs))
        with progress_disabled_ctx(self.model) as model:
            preds, _ = model.get_preds(ds_type=DatasetType.Test)
        
        if self.problem_type == REGRESSION:
            return preds.numpy().reshape(-1)
        if self.problem_type == BINARY:
            return preds[:, 1].numpy()
        else:
            return preds.numpy()
    
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
        if self.processor is None or self.types_of_features is None or self.feature_arraycol_map is None or self.feature_type_map is None:
            raise ValueError("Need to process training data before test data")
        processed_array = self.processor.transform(df) # 2D numpy array. self.feature_arraycol_map, self.feature_type_map have been previously set while processing training data.
        return TabularNNDataset(processed_array, self.feature_arraycol_map, self.feature_type_map, self.params, labels=labels, is_test=True)
    
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
    
    def save(self): # TODO
        # Export model
        save_pkl.save_with_fn(f'{self.path}{self.model_internals_file_name}', self.model, lambda m, buffer: m.export(buffer, destroy=True))
        self.model = None
        super().save()
    
    @classmethod # TODO
    def load(cls, path: str, reset_paths=False):
        obj = super().load(path)
        obj.model = load_pkl.load_with_fn(f'{obj.path}{obj.model_internals_file_name}', lambda p: load_learner(obj.path, p))
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

"""


## Sratch code

################
# Test data processing:
from autogluon import predict_table_column as task
from tabular.ml.learner.default_learner import DefaultLearner as Learner
from tabular.feature_generators.auto_ml_feature_generator import AutoMLFeatureGenerator

data_dir = '/Users/jonasmue/Documents/Datasets/AdultIncomeOpenMLTask=7592/'
train_file_path = data_dir+'train_adultincomedata.csv'
test_file_path = data_dir+'test_adultincomedata.csv'
savedir = data_dir+'Output/'
label_column = 'class' # name of column containing label to predict
train_data = task.load_data(train_file_path) # returns Pandas object, if user already has pandas object in python, can skip this step
train_data = train_data.head(1000) # subsample for faster demo
print(train_data.head())
predictor = task.fit(train_data=train_data, label=label_column, savedir=savedir, hyperparameter_tune=False) # val=None automatically determines train/val split, otherwise we check to ensure train/val match
trainer = predictor.load_trainer()
example_model = trainer.load_model('NNTabularModel')
# print(trainer.__dict__) # summary of training processes

learner = Learner(path_context=savedir, label=label_column, submission_columns=[], feature_generator=AutoMLFeatureGenerator(), threshold=100)
X, y, X_test, y_test = learner.general_data_processing(X=train_data)
X_train, X_test, y_train, y_test = trainer.generate_train_test_split(X, y) # all pandas dataframes

nn = TabularNeuralNetModel(path='', name='tabularNN', params = {}, problem_type=trainer.problem_type, objective_func=trainer.objective_func)
nn.feature_types_metadata = trainer.feature_types_metadata
X_train = nn.preprocess(X_train)  # TODO from grail: Handle cases where features have been removed due to tuning, currently crashes.
print('features: ', nn.features)
if X_test is not None:
     X_test = nn.preprocess(X_test)

train_dataset = nn.process_data(X_train, y_train, is_test=False) # Dataset object


test_dataset = nn.process_data(X_test, y_test, is_test=True)


#################
# Test Encoders: first need to define encoders in python from categorical_encoders.py!

X = [['Female', 10], ['Male', 1], ['Male', 1], ['Male', 1], ['Male', 3], ['Male', 3], ['Female', 3], ['Female', 2], ['Female', 2]]
Xtest = [['Male', 1], ['Female', 3], ['Female', 2], ['Female', 10],['Female', 10],['Female', 10], ['Ww', 3], ['Male', 300]]

enc = OrdinalMergeRaresHandleUnknownEncoder(max_levels=1)
enc.fit(X)
enc.transform(X)
enc.fit_transform(X)
enc.transform(Xtest)

enc2 = OneHotMergeRaresHandleUnknownEncoder(max_levels=3, sparse=False)
newX = enc2.fit_transform(X)
enc2.transform(Xtest)

unique_category_str = '!missing!'

def _create_preprocessor(feature_types, max_category_levels = 4):
    """ Defines data encoders used to preprocess different data types and creates instance variable which is sklearn ColumnTransformer object """
    continuous_features = feature_types['continuous']
    skewed_features = feature_types['skewed']
    onehot_features = feature_types['onehot']
    embed_features = feature_types['embed']
    language_features = feature_types['language']
    transformers = [] # order of various column transformers in this list is important!
    if len(continuous_features) > 0:
        continuous_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())])
        transformers.append( ('continuous', continuous_transformer, continuous_features) )
    if len(skewed_features) > 0:
        power_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('power', PowerTransformer(method='yeo-johnson')) ])
        transformers.append( ('skewed', power_transformer, skewed_features) )
    if len(onehot_features) > 0:
        onehot_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value=self.nique_category_str)),
            ('onehot', OneHotMergeRaresHandleUnknownEncoder(max_levels=max_category_levels,sparse=False)) ]) # test-time unknown values will be encoded as all zeros vector
        transformers.append( ('onehot', onehot_transformer, onehot_features) )
    if len(embed_features) > 0:
        ordinal_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value=self.unique_category_str)),
            ('ordinal', OrdinalMergeRaresHandleUnknownEncoder(max_levels=max_category_levels)) ]) # returns 0-n when max_category_levels = n-1. category n is reserved for unknown test-time categories.
        transformers.append( ('ordinal', ordinal_transformer, embed_features) )
    if len(language_features) > 0:
        raise NotImplementedError("language_features cannot be used at the moment")
    processor = ColumnTransformer(transformers=transformers) # numeric features are processed in the same order as in numeric_features vector, so feature-names remain the same.
    return processor


df = train_data.iloc[:8,:5]
continuous_features = ['education-num']
skewed_features = ['fnlwgt','age']
onehot_features = ['workclass','education']
embed_features = []
language_features = []
feature_types = {'continuous': continuous_features, 'skewed': skewed_features, 'onehot': onehot_features, 
                         'embed': embed_features, 'language': language_features}

preprocessor = _create_preprocessor(feature_types)

processed_df = preprocessor.fit_transform(df)

processed_df.columns = _get_feature_names(preprocessor,df,continuous_features,skewed_features,onehot_features,embed_features,language_features)
