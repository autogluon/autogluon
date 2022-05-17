""" MXNet neural networks for tabular data containing numerical, categorical, and text fields.
    First performs neural network specific pre-processing of the data.
    Contains separate input modules which are applied to different columns of the data depending on the type of values they contain:
    - Numeric columns are pased through single Dense layer (binary categorical variables are treated as numeric)
    - Categorical columns are passed through separate Embedding layers
    Vectors produced by different input layers are then concatenated and passed to multi-layer MLP model with problem_type determined output layer.
    Hyperparameters are passed as dict params, including options for preprocessing stages.
"""
import json
import logging
import os
import random
import time
import warnings
from collections import OrderedDict

import numpy as np
import pandas as pd

from autogluon.common.features.types import R_BOOL, R_INT, R_FLOAT, R_CATEGORY, S_TEXT_NGRAM, S_TEXT_AS_CATEGORY
from autogluon.core.constants import BINARY, MULTICLASS, REGRESSION, SOFTCLASS
from autogluon.core.utils import try_import_mxboard, try_import_mxnet
from autogluon.core.utils.exceptions import TimeLimitExceeded
from autogluon.core.models.abstract.abstract_nn_model import AbstractNeuralNetworkModel

from ..hyperparameters.parameters import get_default_param
from ..hyperparameters.searchspaces import get_default_searchspace
from ..utils.data_preprocessor import create_preprocessor, get_feature_arraycol_map, get_feature_type_map
from ..utils.nn_architecture_utils import infer_y_range, get_default_layers, default_numeric_embed_dim

warnings.filterwarnings("ignore", module='sklearn.preprocessing')  # sklearn processing n_quantiles warning
logger = logging.getLogger(__name__)
EPS = 1e-10  # small number
_has_warned_mxnet_deprecation = False


# TODO: Gets stuck after infering feature types near infinitely in nyc-jiashenliu-515k-hotel-reviews-data-in-europe dataset, 70 GB of memory, c5.9xlarge
#  Suspect issue is coming from embeddings due to text features with extremely large categorical counts.
class TabularNeuralNetMxnetModel(AbstractNeuralNetworkModel):
    """ Class for neural network models that operate on tabular data.
        These networks use different types of input layers to process different types of data in various columns.

        Attributes:
            _types_of_features (dict): keys = 'continuous', 'skewed', 'onehot', 'embed'; values = column-names of Dataframe corresponding to the features of this type
            feature_arraycol_map (OrderedDict): maps feature-name -> list of column-indices in df corresponding to this feature
        self.feature_type_map (OrderedDict): maps feature-name -> feature_type string (options: 'vector', 'embed')
        processor (sklearn.ColumnTransformer): scikit-learn preprocessor object.

        Note: This model always assumes higher values of self.eval_metric indicate better performance.

    """

    # Constants used throughout this class:
    # model_internals_file_name = 'model-internals.pkl' # store model internals here
    unique_category_str = '!missing!' # string used to represent missing values and unknown categories for categorical features. Should not appear in the dataset
    params_file_name = 'net.params' # Stores parameters of final network
    temp_file_name = 'temp_net.params' # Stores temporary network parameters (eg. during the course of training)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        """
        TabularNeuralNetMxnetModel object.

        Parameters
        ----------
        path (str): file-path to directory where to save files associated with this model
        name (str): name used to refer to this model
        problem_type (str): what type of prediction problem is this model used for
        eval_metric (func): function used to evaluate performance (Note: we assume higher = better)
        hyperparameters (dict): various hyperparameters for neural network and the NN-specific data processing
        features (list): List of predictive features to use, other features are ignored by the model.
        """
        self.feature_arraycol_map = None
        self.feature_type_map = None
        self.features_to_drop = []  # may change between different bagging folds. TODO: consider just removing these from self._features_internal
        self.processor = None  # data processor
        self.summary_writer = None
        self.ctx = None
        self.batch_size = None
        self.num_dataloading_workers = None
        self.num_dataloading_workers_inference = 0
        self.params_post_fit = None
        self.num_net_outputs = None
        self._architecture_desc = None
        self.optimizer = None
        self.verbosity = None

    def _set_default_params(self):
        """ Specifies hyperparameter values to use by default """
        default_params = get_default_param(problem_type=self.problem_type, framework='mxnet')
        for param, val in default_params.items():
            self._set_default_param_value(param, val)

    def _get_default_auxiliary_params(self) -> dict:
        default_auxiliary_params = super()._get_default_auxiliary_params()
        extra_auxiliary_params = dict(
            valid_raw_types=[R_BOOL, R_INT, R_FLOAT, R_CATEGORY],
            ignored_type_group_special=[S_TEXT_NGRAM, S_TEXT_AS_CATEGORY],
        )
        default_auxiliary_params.update(extra_auxiliary_params)
        return default_auxiliary_params

    def _get_default_searchspace(self):
        return get_default_searchspace(problem_type=self.problem_type, framework='mxnet')

    def set_net_defaults(self, train_dataset, params):
        """ Sets dataset-adaptive default values to use for our neural network """
        if self.problem_type in [MULTICLASS, SOFTCLASS]:
            self.num_net_outputs = train_dataset.num_classes
        elif self.problem_type == REGRESSION:
            self.num_net_outputs = 1
            if params['y_range'] is None:
                params['y_range'] = infer_y_range(y_vals=train_dataset.dataset._data[train_dataset.label_index].asnumpy(), y_range_extend=params['y_range_extend'])
        elif self.problem_type == BINARY:
            self.num_net_outputs = 2
        else:
            raise ValueError("unknown problem_type specified: %s" % self.problem_type)

        if params['layers'] is None:  # Use default choices for MLP architecture
            params['layers'] = get_default_layers(problem_type=self.problem_type, num_net_outputs=self.num_net_outputs, max_layer_width=params['max_layer_width'])

        if train_dataset.has_vector_features() and params['numeric_embed_dim'] is None:  # Use default choices for numeric embedding size
            params['numeric_embed_dim'] = default_numeric_embed_dim(train_dataset=train_dataset, max_layer_width=params['max_layer_width'], first_layer_width=params['layers'][0])
        return

    def _fit(self, X, y, X_val=None, y_val=None,
             time_limit=None, sample_weight=None, num_cpus=1, num_gpus=0, reporter=None, **kwargs):
        """ X (pd.DataFrame): training data features (not necessarily preprocessed yet)
            X_val (pd.DataFrame): test data features (should have same column names as Xtrain)
            y (pd.Series):
            y_val (pd.Series): are pandas Series
            kwargs: Can specify amount of compute resources to utilize (num_cpus, num_gpus).
        """
        start_time = time.time()
        try_import_mxnet()
        import mxnet as mx
        self.verbosity = kwargs.get('verbosity', 2)
        global _has_warned_mxnet_deprecation
        if not _has_warned_mxnet_deprecation:
            _has_warned_mxnet_deprecation = True
            logger.log(30, '\tWARNING: TabularNeuralNetMxnetModel (alias "NN" & "NN_MXNET") has been deprecated in v0.4.0.\n'
                           '\t\tStarting in v0.5.0, calling TabularNeuralNetMxnetModel will raise an exception.\n'
                           '\t\tConsider instead using TabularNeuralNetTorchModel via "NN_TORCH".')

        if sample_weight is not None:  # TODO: support
            logger.log(15, "sample_weight not yet supported for TabularNeuralNetModel, this model will ignore them in training.")

        params = self._get_model_params()
        if num_cpus is not None:
            self.num_dataloading_workers = max(1, int(num_cpus/2.0))
        else:
            self.num_dataloading_workers = 1
        if self.num_dataloading_workers == 1:
            self.num_dataloading_workers = 0  # 0 is always faster and uses less memory than 1
        self.batch_size = params['batch_size']
        train_dataset, val_dataset = self.generate_datasets(X=X, y=y, params=params, X_val=X_val, y_val=y_val)
        logger.log(15, "Training data for neural network has: %d examples, %d features (%d vector, %d embedding)" %
                   (train_dataset.num_examples, train_dataset.num_features, len(train_dataset.feature_groups['vector']), len(train_dataset.feature_groups['embed'])
                  ))
        # self._save_preprocessor()  # TODO: should save these things for hyperparam tunning. Need one HP tuner for network-specific HPs, another for preprocessing HPs.

        if num_gpus is not None and num_gpus >= 1:
            self.ctx = mx.gpu()  # Currently cannot use more than 1 GPU
        else:
            self.ctx = mx.cpu()
        self.get_net(train_dataset, params=params)

        if time_limit is not None:
            time_elapsed = time.time() - start_time
            time_limit_orig = time_limit
            time_limit = time_limit - time_elapsed
            if time_limit <= time_limit_orig * 0.4:  # if 60% of time was spent preprocessing, likely not enough time to train model
                raise TimeLimitExceeded

        self.train_net(train_dataset=train_dataset, params=params, val_dataset=val_dataset, initialize=True, setup_trainer=True, time_limit=time_limit, reporter=reporter)
        self.params_post_fit = params
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
                model.path = original_path
        """

    def get_net(self, train_dataset, params):
        """ Creates a Gluon neural net and context for this dataset.
            Also sets up trainer/optimizer as necessary.
        """
        from .embednet import EmbedNet
        self.set_net_defaults(train_dataset, params)
        self.model = EmbedNet(train_dataset=train_dataset, params=params, num_net_outputs=self.num_net_outputs, ctx=self.ctx)

        # TODO: Below should not occur until at time of saving
        if not os.path.exists(self.path):
            os.makedirs(self.path)

    def train_net(self, train_dataset, params, val_dataset=None, initialize=True, setup_trainer=True, time_limit=None, reporter=None):
        """ Trains neural net on given train dataset, early stops based on test_dataset.
            Args:
                train_dataset (TabularNNDataset): training data used to learn network weights
                val_dataset (TabularNNDataset): validation data used for hyperparameter tuning
                initialize (bool): set = False to continue training of a previously trained model, otherwise initializes network weights randomly
                setup_trainer (bool): set = False to reuse the same trainer from a previous training run, otherwise creates new trainer from scratch
        """
        start_time = time.time()
        import mxnet as mx
        logger.log(15, "Training neural network for up to %s epochs..." % params['num_epochs'])
        seed_value = params.get('seed_value', 0)
        if seed_value is not None:  # Set seeds
            random.seed(seed_value)
            np.random.seed(seed_value)
            mx.random.seed(seed_value)
        if initialize:  # Initialize the weights of network
            logging.debug("initializing neural network...")
            self.model.collect_params().initialize(ctx=self.ctx)
            self.model.hybridize()
            logging.debug("initialized")
        if setup_trainer:
            # Also setup mxboard to monitor training if visualizer has been specified:
            visualizer = self.params_aux.get('visualizer', 'none')
            if visualizer == 'tensorboard' or visualizer == 'mxboard':
                try_import_mxboard()
                from mxboard import SummaryWriter
                self.summary_writer = SummaryWriter(logdir=self.path, flush_secs=5, verbose=False)
            self.optimizer = self.setup_trainer(params=params, train_dataset=train_dataset)
        best_val_metric = -np.inf  # higher = better
        val_metric = None
        best_val_epoch = 0
        val_improve_epoch = 0  # most recent epoch where validation-score strictly improved
        num_epochs = params['num_epochs']
        if val_dataset is not None:
            y_val = val_dataset.get_labels()
        else:
            y_val = None

        if params['loss_function'] is None:
            if self.problem_type == REGRESSION:
                params['loss_function'] = mx.gluon.loss.L1Loss()
            elif self.problem_type == SOFTCLASS:
                params['loss_function'] = mx.gluon.loss.SoftmaxCrossEntropyLoss(sparse_label=False, from_logits=self.model.from_logits)
            else:
                params['loss_function'] = mx.gluon.loss.SoftmaxCrossEntropyLoss(from_logits=self.model.from_logits)

        loss_func = params['loss_function']
        epochs_wo_improve = params['epochs_wo_improve']
        loss_scaling_factor = 1.0  # we divide loss by this quantity to stabilize gradients

        rescale_losses = {mx.gluon.loss.L1Loss: 'std', mx.gluon.loss.HuberLoss: 'std', mx.gluon.loss.L2Loss: 'var'}  # dict of loss names where we should rescale loss, value indicates how to rescale.
        loss_torescale = [key for key in rescale_losses if isinstance(loss_func, key)]
        if loss_torescale:
            loss_torescale = loss_torescale[0]
            if rescale_losses[loss_torescale] == 'std':
                loss_scaling_factor = np.std(train_dataset.get_labels())/5.0 + EPS  # std-dev of labels
            elif rescale_losses[loss_torescale] == 'var':
                loss_scaling_factor = np.var(train_dataset.get_labels())/5.0 + EPS  # variance of labels
            else:
                raise ValueError("Unknown loss-rescaling type %s specified for loss_func==%s" % (rescale_losses[loss_torescale], loss_func))

        if self.verbosity <= 1:
            verbose_eval = -1  # Print losses every verbose epochs, Never if -1
        elif self.verbosity == 2:
            verbose_eval = 50
        elif self.verbosity == 3:
            verbose_eval = 10
        else:
            verbose_eval = 1

        net_filename = self.path + self.temp_file_name
        if num_epochs == 0:  # use dummy training loop that stops immediately (useful for using NN just for data preprocessing / debugging)
            logger.log(20, "Not training Neural Net since num_epochs == 0.  Neural network architecture is:")
            for batch_idx, data_batch in enumerate(train_dataset.dataloader):
                data_batch = train_dataset.format_batch_data(data_batch, self.ctx)
                with mx.autograd.record():
                    output = self.model(data_batch)
                    labels = data_batch['label']
                    loss = loss_func(output, labels) / loss_scaling_factor
                    # print(str(mx.nd.mean(loss).asscalar()), end="\r")  # prints per-batch losses
                loss.backward()
                self.optimizer.step(labels.shape[0])
                if batch_idx > 0:
                    break
            self.model.save_parameters(net_filename)
            logger.log(15, "untrained Neural Net saved to file")
            return

        start_fit_time = time.time()
        if time_limit is not None:
            time_limit = time_limit - (start_fit_time - start_time)

        # Training Loop:
        for e in range(num_epochs):
            if e == 0:  # special actions during first epoch:
                logger.log(15, "Neural network architecture:")
                logger.log(15, str(self.model))
            cumulative_loss = 0
            for batch_idx, data_batch in enumerate(train_dataset.dataloader):
                data_batch = train_dataset.format_batch_data(data_batch, self.ctx)
                with mx.autograd.record():
                    output = self.model(data_batch)
                    labels = data_batch['label']
                    loss = loss_func(output, labels) / loss_scaling_factor
                    # print(str(mx.nd.mean(loss).asscalar()), end="\r")  # prints per-batch losses
                loss.backward()
                self.optimizer.step(labels.shape[0])
                cumulative_loss += loss.sum()
            train_loss = cumulative_loss/float(train_dataset.num_examples)  # training loss this epoch
            if val_dataset is not None:
                # FIXME: Switch to adaptive ES
                val_metric = self.score(X=val_dataset, y=y_val, metric=self.stopping_metric)
                if np.isnan(val_metric):
                    if e == 0:
                        raise RuntimeError("NaNs encountered in TabularNeuralNetModel training. Features/labels may be improperly formatted or NN weights may have diverged.")
                    else:
                        logger.warning("Warning: NaNs encountered in TabularNeuralNetModel training. Reverting model to last checkpoint without NaNs.")
                        break
                if (val_metric >= best_val_metric) or (e == 0):
                    if val_metric > best_val_metric:
                        val_improve_epoch = e
                    best_val_metric = val_metric
                    best_val_epoch = e
                    # Until functionality is added to restart training from a particular epoch, there is no point in saving params without test_dataset
                    self.model.save_parameters(net_filename)
            else:
                best_val_epoch = e
            if val_dataset is not None:
                if verbose_eval > 0 and e % verbose_eval == 0:
                    logger.log(15, "Epoch %s.  Train loss: %s, Val %s: %s" %
                      (e, train_loss.asscalar(), self.stopping_metric.name, val_metric))
                if self.summary_writer is not None:
                    self.summary_writer.add_scalar(tag='val_'+self.stopping_metric.name,
                                                   value=val_metric, global_step=e)
            else:
                if verbose_eval > 0 and e % verbose_eval == 0:
                    logger.log(15, "Epoch %s.  Train loss: %s" % (e, train_loss.asscalar()))
            if self.summary_writer is not None:
                self.summary_writer.add_scalar(tag='train_loss', value=train_loss.asscalar(), global_step=e)  # TODO: do we want to keep mxboard support?
            if reporter is not None:
                # TODO: Ensure reporter/scheduler properly handle None/nan values after refactor
                if val_dataset is not None and (not np.isnan(val_metric)):  # TODO: This might work without the if statement
                    # epoch must be number of epochs done (starting at 1)
                    reporter(epoch=e + 1,
                             validation_performance=val_metric,  # Higher val_metric = better
                             train_loss=float(train_loss.asscalar()),
                             eval_metric=self.eval_metric.name,
                             greater_is_better=self.eval_metric.greater_is_better)
            if e - val_improve_epoch > epochs_wo_improve:
                break  # early-stop if validation-score hasn't strictly improved in `epochs_wo_improve` consecutive epochs
            if time_limit is not None:
                time_elapsed = time.time() - start_fit_time
                time_epoch_average = time_elapsed / (e+1)
                time_left = time_limit - time_elapsed
                if time_left < time_epoch_average:
                    logger.log(20, f"\tRan out of time, stopping training early. (Stopping on epoch {e})")
                    break

        if val_dataset is not None:
            self.model.load_parameters(net_filename)  # Revert back to best model
            try:
                os.remove(net_filename)
            except FileNotFoundError:
                pass
        if val_dataset is None:
            logger.log(15, "Best model found in epoch %d" % best_val_epoch)
        else:  # evaluate one final time:
            final_val_metric = self.score(X=val_dataset, y=y_val, metric=self.stopping_metric)
            if np.isnan(final_val_metric):
                final_val_metric = -np.inf
            logger.log(15, "Best model found in epoch %d. Val %s: %s" %
                  (best_val_epoch, self.stopping_metric.name, final_val_metric))
        self.params_trained['num_epochs'] = best_val_epoch + 1
        return

    def _predict_proba(self, X, **kwargs):
        """ To align predict with abstract_model API.
            Preprocess here only refers to feature processing steps done by all AbstractModel objects,
            not tabularNN-specific preprocessing steps.
            If X is not DataFrame but instead TabularNNDataset object, we can still produce predictions,
            but cannot use preprocess in this case (needs to be already processed).
        """
        from .tabular_nn_dataset import TabularNNDataset
        if isinstance(X, TabularNNDataset):
            return self._predict_tabular_data(new_data=X, process=False, predict_proba=True)
        elif isinstance(X, pd.DataFrame):
            X = self.preprocess(X, **kwargs)
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
        from .tabular_nn_dataset import TabularNNDataset
        import mxnet as mx
        if process:
            new_data = self.process_test_data(new_data, batch_size=self.batch_size, num_dataloading_workers=self.num_dataloading_workers_inference, labels=None)
        if not isinstance(new_data, TabularNNDataset):
            raise ValueError("new_data must of of type TabularNNDataset if process=False")
        if self.problem_type == REGRESSION or not predict_proba:
            preds = mx.nd.zeros((new_data.num_examples,1))
        else:
            preds = mx.nd.zeros((new_data.num_examples, self.num_net_outputs))
        i = 0
        for batch_idx, data_batch in enumerate(new_data.dataloader):
            data_batch = new_data.format_batch_data(data_batch, self.ctx)
            preds_batch = self.model(data_batch)
            batch_size = len(preds_batch)
            if self.problem_type != REGRESSION:
                if not predict_proba: # need to take argmax
                    preds_batch = mx.nd.argmax(preds_batch, axis=1, keepdims=True)
                else: # need to take softmax
                    preds_batch = mx.nd.softmax(preds_batch, axis=1)
            preds[i:(i+batch_size)] = preds_batch
            i = i+batch_size
        if self.problem_type == REGRESSION or not predict_proba:
            return preds.asnumpy().flatten()  # return 1D numpy array
        elif self.problem_type == BINARY and predict_proba:
            return preds[:,1].asnumpy()  # for binary problems, only return P(Y==+1)

        return preds.asnumpy()  # return 2D numpy array

    def generate_datasets(self, X, y, params, X_val=None, y_val=None):
        impute_strategy = params['proc.impute_strategy']
        max_category_levels = params['proc.max_category_levels']
        skew_threshold = params['proc.skew_threshold']
        embed_min_categories = params['proc.embed_min_categories']
        use_ngram_features = params['use_ngram_features']

        from .tabular_nn_dataset import TabularNNDataset
        if isinstance(X, TabularNNDataset):
            train_dataset = X
        else:
            X = self.preprocess(X)
            train_dataset = self.process_train_data(
                df=X, labels=y, batch_size=self.batch_size, num_dataloading_workers=self.num_dataloading_workers,
                impute_strategy=impute_strategy, max_category_levels=max_category_levels, skew_threshold=skew_threshold, embed_min_categories=embed_min_categories, use_ngram_features=use_ngram_features,
            )
        if X_val is not None:
            if isinstance(X_val, TabularNNDataset):
                val_dataset = X_val
            else:
                X_val = self.preprocess(X_val)
                val_dataset = self.process_test_data(df=X_val, labels=y_val, batch_size=self.batch_size, num_dataloading_workers=self.num_dataloading_workers_inference)
        else:
            val_dataset = None
        return train_dataset, val_dataset

    def process_test_data(self, df, batch_size, num_dataloading_workers, labels=None):
        """ Process train or test DataFrame into a form fit for neural network models.
        Args:
            df (pd.DataFrame): Data to be processed (X)
            labels (pd.Series): labels to be processed (y)
            test (bool): Is this test data where each datapoint should be processed separately using predetermined preprocessing steps.
                         Otherwise preprocessor uses all data to determine propreties like best scaling factors, number of categories, etc.
        Returns:
            Dataset object
        """
        from .tabular_nn_dataset import TabularNNDataset
        warnings.filterwarnings("ignore", module='sklearn.preprocessing') # sklearn processing n_quantiles warning
        if labels is not None and len(labels) != len(df):
            raise ValueError("Number of examples in Dataframe does not match number of labels")
        if (self.processor is None or self._types_of_features is None
           or self.feature_arraycol_map is None or self.feature_type_map is None):
            raise ValueError("Need to process training data before test data")
        if self.features_to_drop:
            drop_cols = [col for col in df.columns if col in self.features_to_drop]
            if drop_cols:
                df = df.drop(columns=drop_cols)

        df = self.processor.transform(df) # 2D numpy array. self.feature_arraycol_map, self.feature_type_map have been previously set while processing training data.
        return TabularNNDataset(df, self.feature_arraycol_map, self.feature_type_map,
                                batch_size=batch_size, num_dataloading_workers=num_dataloading_workers,
                                problem_type=self.problem_type, labels=labels, is_test=True)

    def process_train_data(self, df, batch_size, num_dataloading_workers, impute_strategy, max_category_levels, skew_threshold, embed_min_categories, use_ngram_features, labels):
        """ Preprocess training data and create self.processor object that can be used to process future data.
            This method should only be used once per TabularNeuralNetModel object, otherwise will produce Warning.

        # TODO no label processing for now
        # TODO: add time/ngram features
        # TODO: no filtering of data-frame columns based on statistics, e.g. categorical columns with all unique variables or zero-variance features.
                This should be done in default_learner class for all models not just TabularNeuralNetModel...
        """
        from .tabular_nn_dataset import TabularNNDataset
        warnings.filterwarnings("ignore", module='sklearn.preprocessing')  # sklearn processing n_quantiles warning
        if labels is None:
            raise ValueError("Attempting process training data without labels")
        if len(labels) != len(df):
            raise ValueError("Number of examples in Dataframe does not match number of labels")

        self._types_of_features, df = self._get_types_of_features(df, skew_threshold=skew_threshold, embed_min_categories=embed_min_categories, use_ngram_features=use_ngram_features)  # dict with keys: : 'continuous', 'skewed', 'onehot', 'embed', values = column-names of df
        logger.log(15, "Tabular Neural Network treats features as the following types:")
        logger.log(15, json.dumps(self._types_of_features, indent=4))
        logger.log(15, "\n")
        if self.processor is not None:
            Warning("Attempting to process training data for TabularNeuralNetModel, but previously already did this.")
        self.processor = create_preprocessor(impute_strategy=impute_strategy, max_category_levels=max_category_levels, unique_category_str=self.unique_category_str, continuous_features=self._types_of_features['continuous'],
                                   skewed_features=self._types_of_features['skewed'], onehot_features=self._types_of_features['onehot'], embed_features=self._types_of_features['embed'], bool_features=self._types_of_features['bool'])
        df = self.processor.fit_transform(df) # 2D numpy array
        self.feature_arraycol_map = get_feature_arraycol_map(processor=self.processor, max_category_levels=max_category_levels)  # OrderedDict of feature-name -> list of column-indices in df corresponding to this feature
        num_array_cols = np.sum([len(self.feature_arraycol_map[key]) for key in self.feature_arraycol_map])  # should match number of columns in processed array
        if num_array_cols != df.shape[1]:
            raise ValueError("Error during one-hot encoding data processing for neural network. Number of columns in df array does not match feature_arraycol_map.")

        self.feature_type_map = get_feature_type_map(feature_arraycol_map=self.feature_arraycol_map, types_of_features=self._types_of_features)  # OrderedDict of feature-name -> feature_type string (options: 'vector', 'embed')
        return TabularNNDataset(df, self.feature_arraycol_map, self.feature_type_map,
                                batch_size=batch_size, num_dataloading_workers=num_dataloading_workers,
                                problem_type=self.problem_type, labels=labels, is_test=False)

    def setup_trainer(self, params, train_dataset=None):
        """ Set up optimizer needed for training.
            Network must first be initialized before this.
        """
        import mxnet as mx
        optimizer_opts = {'learning_rate': params['learning_rate'], 'wd': params['weight_decay'], 'clip_gradient': params['clip_gradient']}
        if 'lr_scheduler' in params and params['lr_scheduler'] is not None:
            if train_dataset is None:
                raise ValueError("train_dataset cannot be None when lr_scheduler is specified.")
            base_lr = params.get('base_lr', 1e-6)
            target_lr = params.get('target_lr', 1.0)
            warmup_epochs = params.get('warmup_epochs', 10)
            lr_decay = params.get('lr_decay', 0.1)
            lr_mode = params['lr_scheduler']
            num_batches = train_dataset.num_examples // params['batch_size']
            lr_decay_epoch = [max(warmup_epochs, int(params['num_epochs']/3)), max(warmup_epochs+1, int(params['num_epochs']/2)),
                              max(warmup_epochs+2, int(2*params['num_epochs']/3))]
            from .lr_scheduler import LRSequential, LRScheduler
            lr_scheduler = LRSequential([
                LRScheduler('linear', base_lr=base_lr, target_lr=target_lr, nepochs=warmup_epochs, iters_per_epoch=num_batches),
                LRScheduler(lr_mode, base_lr=target_lr, target_lr=base_lr, nepochs=params['num_epochs'] - warmup_epochs,
                            iters_per_epoch=num_batches, step_epoch=lr_decay_epoch, step_factor=lr_decay, power=2)
            ])
            optimizer_opts['lr_scheduler'] = lr_scheduler
        if params['optimizer'] == 'sgd':
            if 'momentum' in params:
                optimizer_opts['momentum'] = params['momentum']
            optimizer = mx.gluon.Trainer(self.model.collect_params(), 'sgd', optimizer_opts)
        elif params['optimizer'] == 'adam':  # TODO: Can we try AdamW?
            optimizer = mx.gluon.Trainer(self.model.collect_params(), 'adam', optimizer_opts)
        else:
            raise ValueError("Unknown optimizer specified: %s" % params['optimizer'])
        return optimizer

    def get_info(self):
        info = super().get_info()
        info['hyperparameters_post_fit'] = self.params_post_fit
        return info

    def reduce_memory_size(self, remove_fit=True, requires_save=True, **kwargs):
        super().reduce_memory_size(remove_fit=remove_fit, requires_save=requires_save, **kwargs)
        if remove_fit and requires_save:
            self.optimizer = None

    def _get_default_stopping_metric(self):
        return self.eval_metric

    def save(self, path: str = None, verbose=True) -> str:
        if self.model is not None:
            self._architecture_desc = self.model.architecture_desc
        temp_model = self.model
        temp_sw = self.summary_writer
        self.model = None
        self.summary_writer = None
        path_final = super().save(path=path, verbose=verbose)
        self.model = temp_model
        self.summary_writer = temp_sw
        self._architecture_desc = None

        # Export model
        if self.model is not None:
            params_filepath = path_final + self.params_file_name
            # TODO: Don't use os.makedirs here, have save_parameters function in tabular_nn_model that checks if local path or S3 path
            os.makedirs(os.path.dirname(path_final), exist_ok=True)
            self.model.save_parameters(params_filepath)
        return path_final

    @classmethod
    def load(cls, path: str, reset_paths=True, verbose=True):
        model: TabularNeuralNetMxnetModel = super().load(path=path, reset_paths=reset_paths, verbose=verbose)
        if model._architecture_desc is not None:
            from .embednet import EmbedNet
            model.model = EmbedNet(architecture_desc=model._architecture_desc, ctx=model.ctx)  # recreate network from architecture description
            model._architecture_desc = None
            # TODO: maybe need to initialize/hybridize?
            model.model.load_parameters(model.path + model.params_file_name, ctx=model.ctx)
            model.summary_writer = None
        return model

    def _more_tags(self):
        # `can_refit_full=True` because num_epochs is communicated at end of `_fit`: `self.params_trained['num_epochs'] = best_val_epoch + 1`
        return {'can_refit_full': True}



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
