""" MXNet neural networks for tabular data containing numerical, categorical, and text fields.
    First performs neural network specific pre-processing of the data.
    Contains separate input modules which are applied to different columns of the data depending on the type of values they contain:
    - Numeric columns are pased through single Dense layer (binary categorical variables are treated as numeric)
    - Categorical columns are passed through separate Embedding layers
    - Text columns are passed through separate LanguageModel layers
    Vectors produced by different input layers are then concatenated and passed to multi-layer MLP model with problem_type determined output layer.
    Hyperparameters are passed as dict params, including options for preprocessing stages.
"""
import random, json, time, os, logging, warnings
from collections import OrderedDict
import numpy as np
import pandas as pd
import mxnet as mx
from mxnet import nd, autograd, gluon
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, QuantileTransformer  # PowerTransformer

from ......core import Space
from ......utils import try_import_mxboard
from ......task.base import BasePredictor
from ....utils.loaders import load_pkl
from ..abstract.abstract_model import AbstractModel, fixedvals_from_searchspaces
from ....utils.savers import save_pkl
from ...constants import BINARY, MULTICLASS, REGRESSION
from .categorical_encoders import OneHotMergeRaresHandleUnknownEncoder, OrdinalMergeRaresHandleUnknownEncoder
from .tabular_nn_dataset import TabularNNDataset
from .embednet import EmbedNet
from .tabular_nn_trial import tabular_nn_trial
from .hyperparameters.parameters import get_default_param
from .hyperparameters.searchspaces import get_default_searchspace

# __all__ = ['TabularNeuralNetModel', 'EPS']

warnings.filterwarnings("ignore", module='sklearn.preprocessing') # sklearn processing n_quantiles warning
logger = logging.getLogger(__name__)
EPS = 10e-8 # small number


# TODO: Gets stuck after infering feature types near infinitely in nyc-jiashenliu-515k-hotel-reviews-data-in-europe dataset, 70 GB of memory, c5.9xlarge
#  Suspect issue is coming from embeddings due to text features with extremely large categorical counts.
class TabularNeuralNetModel(AbstractModel):
    """ Class for neural network models that operate on tabular data. 
        These networks use different types of input layers to process different types of data in various columns.
    
        Attributes:
            types_of_features (dict): keys = 'continuous', 'skewed', 'onehot', 'embed', 'language'; values = column-names of Dataframe corresponding to the features of this type
            feature_arraycol_map (OrderedDict): maps feature-name -> list of column-indices in processed_array corresponding to this feature
        self.feature_type_map (OrderedDict): maps feature-name -> feature_type string (options: 'vector', 'embed', 'language')
        processor (sklearn.ColumnTransformer): scikit-learn preprocessor object.
        
        Note: This model always assumes higher values of self.objective_func indicate better performance.
        
    """
    
    # Constants used throughout this class:
    # model_internals_file_name = 'model-internals.pkl' # store model internals here
    unique_category_str = '!missing!' # string used to represent missing values and unknown categories for categorical features. Should not appear in the dataset
    # TODO: remove: metric_map = {REGRESSION: 'Rsquared', BINARY: 'accuracy', MULTICLASS: 'accuracy'}  # string used to represent different evaluation metrics. metric_map[self.problem_type] produces str corresponding to metric used here.
    # TODO: should be using self.objective_func as the metric of interest. Should have method: get_metric_name(self.objective_func)
    rescale_losses = {gluon.loss.L1Loss:'std', gluon.loss.HuberLoss:'std', gluon.loss.L2Loss:'var'} # dict of loss names where we should rescale loss, value indicates how to rescale. Call self.loss_func.name
    model_file_name = 'tabularNN.pkl'
    params_file_name = 'net.params' # Stores parameters of final network
    temp_file_name = 'temp_net.params' # Stores temporary network parameters (eg. during the course of training)
    
    def __init__(self, path: str, name: str, problem_type: str, objective_func, hyperparameters=None, features=None):
        super().__init__(path=path, name=name, problem_type=problem_type, objective_func=objective_func, hyperparameters=hyperparameters, features=features)
        """
        TabularNeuralNetModel object.
        
        Parameters
        ----------
        path (str): file-path to directory where to save files associated with this model
        name (str): name used to refer to this model
        problem_type (str): what type of prediction problem is this model used for
        objective_func (func): function used to evaluate performance (Note: we assume higher = better) 
        hyperparameters (dict): various hyperparameters for neural network and the NN-specific data processing
        features (list): List of predictive features to use, other features are ignored by the model.
        """
        self.problem_type = problem_type
        self.objective_func = objective_func
        self.eval_metric_name = self.objective_func.name
        self.feature_types_metadata = None
        self.types_of_features = None
        self.feature_arraycol_map = None
        self.feature_type_map = None
        self.processor = None # data processor
        self.summary_writer = None
        self.ctx = mx.cpu()

    # TODO: Fix model to not have tabNN in params
    def convert_to_template(self):
        new_model = TabularNeuralNetModel(path=self.path, name=self.name, problem_type=self.problem_type, objective_func=self.objective_func, features=self.features, hyperparameters=self.params)
        new_model.path = self.path
        new_model.params['tabNN'] = None
        return new_model

    def _set_default_params(self):
        """ Specifies hyperparameter values to use by default """
        default_params = get_default_param(self.problem_type)
        for param, val in default_params.items():
            self._set_default_param_value(param, val)

    def set_net_defaults(self, train_dataset):
        """ Sets dataset-adaptive default values to use for our neural network """
        if self.problem_type == MULTICLASS:
            self.num_classes = train_dataset.num_classes
            self.num_net_outputs = self.num_classes
        elif self.problem_type == REGRESSION:
            self.num_net_outputs = 1
            if self.params['y_range'] is None: # Infer default y-range
                y_vals = train_dataset.dataset._data[train_dataset.label_index].asnumpy()
                min_y = float(min(y_vals))
                max_y = float(max(y_vals))
                std_y = np.std(y_vals)
                y_ext = self.params['y_range_extend']*std_y
                if min_y >= 0: # infer y must be nonnegative
                    min_y = max(0, min_y-y_ext)
                else:
                    min_y = min_y-y_ext
                if max_y <= 0: # infer y must be non-positive
                    max_y = min(0, max_y+y_ext)
                else:
                    max_y = max_y+y_ext
                self.params['y_range'] = (min_y, max_y)
        elif self.problem_type == BINARY:
            self.num_classes = 2
            self.num_net_outputs = 2
        
        if self.params['layers'] is None: # Use default choices for MLP architecture
            if self.problem_type == REGRESSION:
                default_layer_sizes = [256, 128] # overall network will have 4 layers. Input layer, 256-unit hidden layer, 128-unit hidden layer, output layer.
            elif self.problem_type == BINARY or self.problem_type == MULTICLASS:
                default_sizes = [256, 128] # will be scaled adaptively
                # base_size = max(1, min(self.num_net_outputs, 20)/2.0) # scale layer width based on number of classes
                base_size = max(1, min(self.num_net_outputs, 100) / 50)  # TODO: Updated because it improved model quality and made training far faster
                default_layer_sizes = [defaultsize*base_size for defaultsize in default_sizes]
            # TODO: This gets really large on 100K+ rows... It takes hours on gpu for nyc-albert: 78 float/int features which get expanded to 1734, it also overfits and maxes accuracy on epoch
            #  LGBM takes 120 seconds on 4 cpu's and gets far better accuracy
            #  Perhaps we should add an order of magnitude to the pre-req with -3, or else scale based on feature count instead of row count.
            # layer_expansion_factor = np.log10(max(train_dataset.num_examples, 1000)) - 2 # scale layers based on num_training_examples
            layer_expansion_factor = 1  # TODO: Hardcoded to 1 because it results in both better model quality and far faster training time
            max_layer_width = self.params['max_layer_width']
            self.params['layers'] = [int(min(max_layer_width, layer_expansion_factor*defaultsize))
                                     for defaultsize in default_layer_sizes]
        
        if train_dataset.has_vector_features() and self.params['numeric_embed_dim'] is None:
            # Use default choices for numeric embedding size
            vector_dim = train_dataset.dataset._data[train_dataset.vectordata_index].shape[1]  # total dimensionality of vector features
            prop_vector_features = train_dataset.num_vector_features() / float(train_dataset.num_features) # Fraction of features that are numeric 
            min_numeric_embed_dim = 32
            max_numeric_embed_dim = self.params['max_layer_width']
            self.params['numeric_embed_dim'] = int(min(max_numeric_embed_dim, max(min_numeric_embed_dim,
                                                    self.params['layers'][0]*prop_vector_features*np.log10(vector_dim+10) )))
        return
    
    def fit(self, X_train, Y_train, X_test=None, Y_test=None, time_limit=None, **kwargs):
        """ X_train (pd.DataFrame): training data features (not necessarily preprocessed yet)
            X_test (pd.DataFrame): test data features (should have same column names as Xtrain)
            Y_train (pd.Series): 
            Y_test (pd.Series): are pandas Series
            kwargs: Can specify amount of compute resources to utilize (num_cpus, num_gpus).
        """
        start_time = time.time()
        self.verbosity = kwargs.get('verbosity', 2)
        self.params = fixedvals_from_searchspaces(self.params)
        if self.feature_types_metadata is None:
            raise ValueError("Trainer class must set feature_types_metadata for this model")
        X_train = self.preprocess(X_train)
        if self.features is None:
            self.features = list(X_train.columns)
        # print('features: ', self.features)
        if 'num_cpus' in kwargs:
            self.params['num_dataloading_workers'] = max(1, int(kwargs['num_cpus']/2.0))
        else:
            self.params['num_dataloading_workers'] = 1
        if 'num_gpus' in kwargs and kwargs['num_gpus'] >= 1: # Currently cannot use >1 GPU
            self.params['ctx'] = mx.gpu() # Currently cannot use more than 1 GPU
        else:
            self.params['ctx'] = mx.cpu()
        train_dataset = self.process_data(X_train, Y_train, is_test=False) # Dataset object
        if X_test is not None:
            X_test = self.preprocess(X_test)
            test_dataset = self.process_data(X_test, Y_test, is_test=True) # Dataset object to use for validation
        else:
            test_dataset = None
        logger.log(15, "Training data for neural network has: %d examples, %d features (%d vector, %d embedding, %d language)" % 
              (train_dataset.num_examples, train_dataset.num_features, 
               len(train_dataset.feature_groups['vector']), len(train_dataset.feature_groups['embed']),
               len(train_dataset.feature_groups['language']) ))
        # train_dataset.save()
        # test_dataset.save()
        # self._save_preprocessor() # TODO: should save these things for hyperparam tunning. Need one HP tuner for network-specific HPs, another for preprocessing HPs.
        
        self.get_net(train_dataset)

        if time_limit:
            time_elapsed = time.time() - start_time
            time_limit = time_limit - time_elapsed

        self.train_net(params=self.params, train_dataset=train_dataset, test_dataset=test_dataset, initialize=True, setup_trainer=True, time_limit=time_limit)
        """
        # TODO: if we don't want to save intermediate network parameters, need to do something like saving in temp directory to clean up after training:
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
    
    def get_net(self, train_dataset):
        """ Creates a Gluon neural net and context for this dataset.
            Also sets up trainer/optimizer as necessary.
        """
        self.set_net_defaults(train_dataset)
        self.ctx = self.params['ctx']
        net = EmbedNet(train_dataset=train_dataset, params=self.params,
                       num_net_outputs=self.num_net_outputs, ctx=self.ctx)
        self.architecture_desc = net.architecture_desc # Description of network architecture
        self.net_filename = self.path + self.temp_file_name
        self.model = net
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        return
    
    def train_net(self, params, train_dataset, test_dataset=None,
                  initialize=True, setup_trainer=True, file_prefix="", time_limit=None):
        """ Trains neural net on given train dataset, early stops based on test_dataset.
            Args:
                params (dict): various hyperparameter values
                train_dataset (TabularNNDataset): training data used to learn network weights
                test_dataset (TabularNNDataset): validation data used for hyperparameter tuning
                initialize (bool): set = False to continue training of a previously trained model, otherwise initializes network weights randomly
                setup_trainer (bool): set = False to reuse the same trainer from a previous training run, otherwise creates new trainer from scratch
                file_prefix (str): prefix to append to all file-names created here. Can use to make sure different trials create different files
        """
        start_time = time.time()
        logger.log(15, "Training neural network for up to %s epochs..." % self.params['num_epochs'])
        seed_value = self.params.get('seed_value')
        if seed_value is not None: # Set seed
            random.seed(seed_value)
            np.random.seed(seed_value)
            mx.random.seed(seed_value)
        if initialize: # Initialize the weights of network
            logging.debug("initializing neural network...")
            self.model.collect_params().initialize(ctx=self.ctx)
            self.model.hybridize()
            logging.debug("initialized")
        if setup_trainer:
            # Also setup mxboard if visualizer has been specified:
            visualizer = self.params.get('visualizer', 'none')
            if visualizer == 'tensorboard' or visualizer == 'mxboard':
                try_import_mxboard()
                from mxboard import SummaryWriter
                self.summary_writer = SummaryWriter(logdir=self.path, flush_secs=5, verbose=False)
            self.setup_trainer()
        best_val_metric = -np.inf # higher = better
        val_metric = None
        best_val_epoch = 0
        best_train_epoch = 0 # epoch with best training loss so far
        best_train_loss = np.inf # smaller = better
        num_epochs = self.params['num_epochs']
        if test_dataset is not None:
            y_test = test_dataset.get_labels()
        else:
            y_test = None
        
        loss_scaling_factor = 1.0 # we divide loss by this quantity to stabilize gradients
        loss_torescale = [key for key in self.rescale_losses if isinstance(self.loss_func, key)]
        if len(loss_torescale) > 0:
            loss_torescale = loss_torescale[0]
            if self.rescale_losses[loss_torescale] == 'std':
                loss_scaling_factor = np.std(train_dataset.get_labels())/5.0 + EPS # std-dev of labels
            elif self.rescale_losses[loss_torescale] == 'var':
                loss_scaling_factor = np.var(train_dataset.get_labels())/5.0 + EPS # variance of labels
            else:
                raise ValueError("Unknown loss-rescaling type %s specified for loss_func==%s" % (self.rescale_losses[loss_torescale],self.loss_func))
        
        if self.verbosity <= 1:
            verbose_eval = -1  # Print losses every verbose epochs, Never if -1
        elif self.verbosity == 2:
            verbose_eval = 50
        elif self.verbosity == 3:
            verbose_eval = 10
        else:
            verbose_eval = 1
        
        # Training Loop:
        for e in range(num_epochs):
            if e == 0: # special actions during first epoch:
                logger.log(15, "Neural network architecture:")
                logger.log(15, str(self.model))  # TODO: remove?
            cumulative_loss = 0
            for batch_idx, data_batch in enumerate(train_dataset.dataloader):
                data_batch = train_dataset.format_batch_data(data_batch, self.ctx)
                with autograd.record():
                    output = self.model(data_batch)
                    labels = data_batch['label']
                    loss = self.loss_func(output, labels) / loss_scaling_factor
                    # print(str(nd.mean(loss).asscalar()), end="\r") # prints per-batch losses
                loss.backward()
                self.optimizer.step(labels.shape[0])
                cumulative_loss += nd.sum(loss).asscalar()
            train_loss = cumulative_loss/float(train_dataset.num_examples) # training loss this epoch
            if test_dataset is not None:
                # val_metric = self.evaluate_metric(test_dataset) # Evaluate after each epoch
                val_metric = self.score(X=test_dataset, y=y_test)
            if test_dataset is None or val_metric >= best_val_metric:  # keep training if score has improved
                best_val_metric = val_metric
                best_val_epoch = e
                self.model.save_parameters(self.net_filename)
            if test_dataset is not None:
                if verbose_eval > 0 and e % verbose_eval == 0:
                    logger.log(15, "Epoch %s.  Train loss: %s, Val %s: %s" %
                      (e, train_loss, self.eval_metric_name, val_metric))
                if self.summary_writer is not None:
                    self.summary_writer.add_scalar(tag='val_'+self.eval_metric_name, 
                                                   value=val_metric, global_step=e)
            else:
                if verbose_eval > 0 and e % verbose_eval == 0:
                    logger.log(15, "Epoch %s.  Train loss: %s" % (e, train_loss))
            if self.summary_writer is not None:
                self.summary_writer.add_scalar(tag='train_loss', value=train_loss, global_step=e)  # TODO: do we want to keep mxboard support?
            if e - best_val_epoch > self.params['epochs_wo_improve']:
                break
            if time_limit:
                time_elapsed = time.time() - start_time
                time_left = time_limit - time_elapsed
                if time_left <= 0:
                    logger.log(20, "\tRan out of time, stopping training early.")
                    break

        self.model.load_parameters(self.net_filename) # Revert back to best model
        if test_dataset is None: # evaluate one final time:
            logger.log(15, "Best model found in epoch %d" % best_val_epoch)
        else:
            final_val_metric = self.score(X=test_dataset, y=y_test)
            logger.log(15, "Best model found in epoch %d. Val %s: %s" %
                  (best_val_epoch, self.eval_metric_name, final_val_metric))
        return

    def evaluate_metric(self, dataset, mx_metric=None):
        """ Evaluates metric on the given dataset (TabularNNDataset object), used for early stopping and to tune hyperparameters.
            If provided, mx_metric must be a function that follows the mxnet.metric API. Higher values = better!
            By default, returns accuracy in the case of classification, R^2 for regression.

            TODO: currently hard-coded metrics used only. Does not respect user-supplied metrics...
        """
        if mx_metric is None:
            if self.problem_type == REGRESSION:
                mx_metric = mx.metric.MSE()
            else:
                mx_metric = mx.metric.Accuracy()
        for batch_idx, data_batch in enumerate(dataset.dataloader):
            data_batch = dataset.format_batch_data(data_batch, self.ctx)
            preds = self.model(data_batch)
            mx_metric.update(preds=preds, labels=data_batch['label']) # argmax not needed, even for classification
        if self.problem_type == REGRESSION:
            y_var = np.var(dataset.dataset._data[dataset.label_index].asnumpy()) + EPS
            return 1.0 - mx_metric.get()[1] / y_var
        else:
            return mx_metric.get()[1] # accuracy

    def predict_proba(self, X, preprocess=True):
        """ To align predict wiht abstract_model API. 
            Preprocess here only refers to feature processing stesp done by all AbstractModel objects, 
            not tabularNN-specific preprocessing steps.
            If X is not DataFrame but instead TabularNNDataset object, we can still produce predictions, 
            but cannot use preprocess in this case (needs to be already processed).
        """
        if isinstance(X, TabularNNDataset):
            return self._predict_tabular_data(new_data=X, process=False, predict_proba=True)
        elif isinstance(X, pd.DataFrame):
            if preprocess:
                X = self.preprocess(X)
            return self._predict_tabular_data(new_data=X, process=True, predict_proba=True)
        else:
            raise ValueError("X must be of type pd.DataFrame or TabularNNDataset, not type: %s" % type(X))

    def _predict_tabular_data(self, new_data, process=True, predict_proba=True):  # TODO ensure API lines up with tabular.Model class.
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
            raise ValueError("new_data must of of type TabularNNDataset if process=False")
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
        warnings.filterwarnings("ignore", module='sklearn.preprocessing') # sklearn processing n_quantiles warning
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
        df = self.ensure_onehot_object(df)
        processed_array = self.processor.transform(df) # 2D numpy array. self.feature_arraycol_map, self.feature_type_map have been previously set while processing training data.
        return TabularNNDataset(processed_array, self.feature_arraycol_map, self.feature_type_map, 
                                self.params, self.problem_type, labels=labels, is_test=True)

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
        df = df[self.features]
        logger.log(15, "AutoGluon Neural Network infers features are of the following types:")
        logger.log(15, json.dumps(self.types_of_features, indent=4))
        logger.log(15, "\n")
        df = self.ensure_onehot_object(df)
        self.processor = self._create_preprocessor()
        processed_array = self.processor.fit_transform(df) # 2D numpy array
        self.feature_arraycol_map = self._get_feature_arraycol_map() # OrderedDict of feature-name -> list of column-indices in processed_array corresponding to this feature
        # print(self.feature_arraycol_map)
        self.feature_type_map = self._get_feature_type_map() # OrderedDict of feature-name -> feature_type string (options: 'vector', 'embed', 'language')
        # print(self.feature_type_map)
        return TabularNNDataset(processed_array, self.feature_arraycol_map, self.feature_type_map,
                                self.params, self.problem_type, labels=labels, is_test=False)

    def setup_trainer(self):
        """ Set up stuff needed for training: 
            optimizer, loss, and summary writer (for mxboard).
            Network must first be initialized before this. 
        """
        optimizer_opts = {'learning_rate': self.params['learning_rate'],  
            'wd': self.params['weight_decay'], 'clip_gradient': self.params['clip_gradient']}
        if self.params['optimizer'] == 'sgd':
            optimizer_opts['momentum'] = self.params['momentum']
            self.optimizer = gluon.Trainer(self.model.collect_params(), 'sgd', optimizer_opts)
        elif self.params['optimizer'] == 'adam':  # TODO: Can we try AdamW?
            self.optimizer = gluon.Trainer(self.model.collect_params(), 'adam', optimizer_opts)
        else:
            raise ValueError("Unknown optimizer specified: %s" % self.params['optimizer'])
        if self.params['loss_function'] is None:
            if self.problem_type == REGRESSION:
                self.params['loss_function'] = gluon.loss.L1Loss()
            else:
                self.params['loss_function'] = gluon.loss.SoftmaxCrossEntropyLoss(from_logits=self.model.from_logits)
        self.loss_func = self.params['loss_function']

    # Helper functions for tabular NN:

    def ensure_onehot_object(self, df):
        """ Converts all numerical one-hot columns to object-dtype. 
            Note: self.types_of_features must already exist! 
        """
        new_df = df.copy() # To avoid SettingWithCopyWarning
        for feature in self.types_of_features['onehot']:
            if df[feature].dtype != 'object':
                new_df.loc[:,feature] = df.loc[:,feature].astype(str)
        return new_df

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
        continuous_featnames = self.__get_feature_type_if_present('float') + self.__get_feature_type_if_present('int') + self.__get_feature_type_if_present('datetime')
        # print("categorical_featnames:", categorical_featnames)
        # print("continuous_featnames:", continuous_featnames)
        language_featnames = [] # TODO: not implemented. This should fetch text features present in the data
        valid_features = categorical_featnames + continuous_featnames + language_featnames
        if len(categorical_featnames) + len(continuous_featnames)\
                + len(language_featnames)\
                != df.shape[1]:
            unknown_features = [feature for feature in df.columns if feature not in valid_features]
            # print('unknown features:', unknown_features)
            df = df.drop(columns=unknown_features)
            self.features = list(df.columns)
            # raise ValueError("unknown feature types present in DataFrame")

        types_of_features = {'continuous': [], 'skewed': [], 'onehot': [], 'embed': [], 'language': []}
        # continuous = numeric features to rescale
        # skewed = features to which we will apply power (ie. log / box-cox) transform before normalization
        # onehot = features to one-hot encode (unknown categories for these features encountered at test-time are encoded as all zeros). We one-hot encode any features encountered that only have two unique values.
        for feature in self.features:
            feature_data = df[feature] # pd.Series
            num_unique_vals = len(feature_data.unique())
            if num_unique_vals == 2:  # will be onehot encoded regardless of proc.embed_min_categories value
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
                ('quantile', QuantileTransformer(output_distribution='normal')) ]) # Or output_distribution = 'uniform'
                # TODO: remove old code: ('power', PowerTransformer(method=self.params['proc.power_transform_method'])) ])
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

    def save(self, file_prefix="", directory = None, return_name=False, verbose=None):
        """ file_prefix (str): Appended to beginning of file-name (does not affect directory in file-path).
            directory (str): if unspecified, use self.path as directory
            return_name (bool): return the file-names corresponding to this save as tuple (model_obj_file, net_params_file)
        """
        if verbose is None:
            verbose = self.verbosity >= 3
        if directory is not None:
            path = directory + file_prefix
        else:
            path = self.path + file_prefix
        
        params_filepath = path + self.params_file_name
        modelobj_filepath = path + self.model_file_name
        if self.model is not None:
            self.model.save_parameters(params_filepath)
        temp_model = self.model
        temp_sw = self.summary_writer
        self.model = None
        self.summary_writer = None
        save_pkl.save(path=modelobj_filepath, object=self, verbose=verbose)
        self.model = temp_model
        self.summary_writer = temp_sw
        if return_name:
            return (modelobj_filepath, params_filepath)

    @classmethod
    def load(cls, path, file_prefix="", reset_paths=False, verbose=True):
        """ file_prefix (str): Appended to beginning of file-name.
            If you want to load files with given prefix, can also pass arg: path = directory+file_prefix
        """
        path = path + file_prefix
        obj = load_pkl.load(path = path + cls.model_file_name, verbose=verbose)
        if reset_paths:
            obj.set_contexts(path)
        obj.model = EmbedNet(architecture_desc=obj.architecture_desc, ctx=obj.ctx) # recreate network from architecture description
        # TODO: maybe need to initialize/hybridize??
        obj.model.load_parameters(path + cls.params_file_name, ctx=obj.ctx)
        obj.summary_writer = None
        return obj

    def hyperparameter_tune(self, X_train, X_test, Y_train, Y_test, scheduler_options, **kwargs):
        """ Performs HPO and sets self.params to best hyperparameter values """
        self.verbosity = kwargs.get('verbosity', 2)
        logger.log(15, "Beginning hyperparameter tuning for Neural Network...")
        self._set_default_searchspace() # changes non-specified default hyperparams from fixed values to search-spaces.
        if self.feature_types_metadata is None:
            raise ValueError("Trainer class must set feature_types_metadata for this model")
        scheduler_func = scheduler_options[0] # Unpack tuple
        scheduler_options = scheduler_options[1]
        if scheduler_func is None or scheduler_options is None:
            raise ValueError("scheduler_func and scheduler_options cannot be None for hyperparameter tuning")
        num_cpus = scheduler_options['resource']['num_cpus']
        num_gpus = scheduler_options['resource']['num_gpus']
        self.params['num_dataloading_workers'] = max(1, int(num_cpus/2.0))
        if num_gpus >= 1:
            self.params['ctx'] = mx.gpu() # Currently cannot use more than 1 GPU until scheduler works
        else:
            self.params['ctx'] = mx.cpu()
        # self.params['ctx'] = mx.cpu() # use this in case embedding layer complains during predict() for HPO with GPU
        
        start_time = time.time()
        X_train = self.preprocess(X_train)
        if self.features is None:
            self.features = list(X_train.columns)
        params_copy = self.params.copy()
        if not np.any([isinstance(params_copy[hyperparam], Space) for hyperparam in params_copy]):
            logger.warning("Warning: Attempting to do hyperparameter optimization without any search space (all hyperparameters are already fixed values)")
        else:
            logger.log(15, "Hyperparameter search space for Neural Network: ")
            for hyperparam in params_copy:
                if isinstance(params_copy[hyperparam], Space):
                    logger.log(15, str(hyperparam)+ ":   "+str(params_copy[hyperparam]))
        directory = self.path # path to directory where all remote workers store things
        train_dataset = self.process_data(X_train, Y_train, is_test=False) # Dataset object
        X_test = self.preprocess(X_test)
        test_dataset = self.process_data(X_test, Y_test, is_test=True) # Dataset object to use for validation
        train_fileprefix = self.path + "train"
        test_fileprefix = self.path + "validation"
        train_dataset.save(file_prefix=train_fileprefix) # TODO: cleanup after HPO?
        test_dataset.save(file_prefix=test_fileprefix)
        tabular_nn_trial.register_args(train_fileprefix=train_fileprefix, test_fileprefix=test_fileprefix,
                                      directory=directory, tabNN=self, **params_copy)
        scheduler = scheduler_func(tabular_nn_trial, **scheduler_options)
        if ('dist_ip_addrs' in scheduler_options) and (len(scheduler_options['dist_ip_addrs']) > 0):
            # This is multi-machine setting, so need to copy dataset to workers:
            logger.log(15, "Uploading preprocessed data to remote workers...")
            scheduler.upload_files([train_fileprefix+TabularNNDataset.DATAOBJ_SUFFIX,
                                train_fileprefix+TabularNNDataset.DATAVALUES_SUFFIX,
                                test_fileprefix+TabularNNDataset.DATAOBJ_SUFFIX,
                                test_fileprefix+TabularNNDataset.DATAVALUES_SUFFIX]) # TODO: currently does not work.
            train_fileprefix = "train"
            test_fileprefix = "validation"
            directory = self.path  # TODO: need to change to path to working directory on every remote machine
            tabular_nn_trial.update(train_fileprefix=train_fileprefix, test_fileprefix=test_fileprefix,
                                   directory=directory)
            logger.log(15, "uploaded")

        scheduler.run()
        scheduler.join_jobs()
        scheduler.get_training_curves(plot=False, use_legend=False)
        # Store results / models from this HPO run:
        best_hp = scheduler.get_best_config() # best_hp only contains searchable stuff
        hpo_results = {'best_reward': scheduler.get_best_reward(),
                       'best_config': best_hp,
                       'total_time': time.time() - start_time,
                       'metadata': scheduler.metadata,
                       'training_history': scheduler.training_history,
                       'config_history': scheduler.config_history,
                       'reward_attr': scheduler._reward_attr,
                       'args': tabular_nn_trial.args
                      }
        hpo_results = BasePredictor._format_results(hpo_results) # store results summarizing HPO for this model
        if ('dist_ip_addrs' in scheduler_options) and (len(scheduler_options['dist_ip_addrs']) > 0):
            raise NotImplementedError("need to fetch model files from remote Workers")
            # TODO: need to handle locations carefully: fetch these 2 files and put into self.path:
            # 1) hpo_results['trial_info'][trial]['metadata']['modelobj_file']
            # 2) hpo_results['trial_info'][trial]['metadata']['netparams_file']
        hpo_models = {} # stores all the model names and file paths to model objects created during this HPO run.
        hpo_model_performances = {}
        for trial in sorted(hpo_results['trial_info'].keys()):
            # TODO: ignore models which were killed early by scheduler (eg. in Hyperband)s
            file_id = "trial_"+str(trial) # unique identifier to files from this trial
            file_prefix = file_id + "_"
            trial_model_name = self.name+"_"+file_id
            trial_model_path = self.path + file_prefix
            hpo_models[trial_model_name] = trial_model_path
            hpo_model_performances[trial_model_name] = hpo_results['trial_info'][trial][scheduler._reward_attr]

        logger.log(15, "Time for Neural Network hyperparameter optimization: %s" % str(hpo_results['total_time']))
        self.params.update(best_hp)
        # TODO: reload model params from best trial? Do we want to save this under cls.model_file as the "optimal model"
        logger.log(15, "Best hyperparameter configuration for Tabular Neural Network: ")
        logger.log(15, str(best_hp))
        return (hpo_models, hpo_model_performances, hpo_results)
        """
        # TODO: do final fit here?
        args.final_fit = True
        model_weights = scheduler.run_with_config(best_config)
        save(model_weights)
        """

    def _set_default_searchspace(self):
        """ Sets up default search space for HPO. Each hyperparameter which user did not specify is converted from 
            default fixed value to default spearch space. 
        """
        search_space = get_default_searchspace(self.problem_type)
        for key in self.nondefault_params: # delete all user-specified hyperparams from the default search space
            _ = search_space.pop(key, None)
        self.params.update(search_space)


""" General TODOs:

- Automatically decrease batch-size if memory issue arises

- Retrain final NN on full dataset (train+val). How to ensure stability here?
- OrdinalEncoder class in sklearn currently cannot handle rare categories or unknown ones at test-time, so we have created our own Encoder in category_encoders.py
There is open PR in sklearn to address this: https://github.com/scikit-learn/scikit-learn/pull/13833/files
Currently, our code uses category_encoders package (BSD license) instead: https://github.com/scikit-learn-contrib/categorical-encoding
Once PR is merged into sklearn, may want to switch: category_encoders.Ordinal -> sklearn.preprocessing.OrdinalEncoder in preprocess_train_data()

- Save preprocessed data so that we can do HPO of neural net hyperparameters more efficiently, while also doing HPO of preprocessing hyperparameters?
      Naive full HPO method requires redoing preprocessing in each trial even if we did not change preprocessing hyperparameters.
      Alternative is we save each proprocessed dataset & corresponding TabularNeuralNetModel object with its unique param names in the file. Then when we try a new HP-config, we first try loading from file if one exists.

"""
