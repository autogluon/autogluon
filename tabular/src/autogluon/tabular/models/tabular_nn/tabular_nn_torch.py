import json
import logging
import os
import random
import time
import warnings

import torch
import torch.nn as nn
import numpy as np
import pandas as pd

from autogluon.core import Space
from autogluon.core.constants import QUANTILE
from autogluon.core.utils import try_import_torch
from autogluon.core.models.abstract import model_trial
from autogluon.core.utils.exceptions import TimeLimitExceeded
from autogluon.core import args
from autogluon.core.models.abstract.abstract_model import AbstractNeuralNetworkModel
from .embednet import getEmbedSizes
from .tabular_nn_model import TabularNeuralNetModel
from ..utils import fixedvals_from_searchspaces


logger = logging.getLogger(__name__)


class TabularNeuralQuantileModel(TabularNeuralNetModel):
    """
    Class for neural network models that operate on tabular data for multi-quantile prediction based on PyTorch.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.problem_type != QUANTILE:
            raise ValueError("This neural network is only available for quantile regression")

    def set_net_defaults(self, train_dataset, params):
        """ Sets dataset-adaptive default values to use for our neural network """
        # infer default y-range
        if params['y_range'] is None:
            y_vals = train_dataset.data_list[train_dataset.label_index]
            min_y = float(np.min(y_vals))
            max_y = float(np.max(y_vals))
            std_y = np.std(y_vals)
            y_ext = params['y_range_extend'] * std_y

            # infer y must be non-negative
            if min_y >= 0:
                min_y = max(0, min_y - y_ext)
            else:
                min_y = min_y - y_ext

            # infer y must be non-positive
            if max_y <= 0:
                max_y = min(0, max_y + y_ext)
            else:
                max_y = max_y+y_ext
            params['y_range'] = (min_y, max_y)
        return

    def _fit(self,
             X,
             y,
             X_val=None,
             y_val=None,
             time_limit=None,
             sample_weight=None,
             num_cpus=1,
             num_gpus=0,
             reporter=None,
             **kwargs):
        start_time = time.time()
        try_import_torch()
        import torch
        self.verbosity = kwargs.get('verbosity', 2)
        if sample_weight is not None:  # TODO: support
            logger.log(15, "sample_weight not yet supported for TabularNeuralQuantileModel,"
                           " this model will ignore them in training.")
        params = self.params.copy()
        params = fixedvals_from_searchspaces(params)
        if self.feature_metadata is None:
            raise ValueError("Trainer class must set feature_metadata for this model")
        if num_cpus is not None:
            self.num_dataloading_workers = max(1, int(num_cpus/2.0))
        else:
            self.num_dataloading_workers = 1
        if self.num_dataloading_workers == 1:
            self.num_dataloading_workers = 0  # 0 is always faster and uses less memory than 1
        self.num_dataloading_workers = 0
        self.max_batch_size = params['max_batch_size']
        if isinstance(X, TabularPyTorchDataset):
            self.batch_size = min(int(2 ** (3 + np.floor(np.log10(len(X))))), self.max_batch_size)
        else:
            self.batch_size = min(int(2 ** (3 + np.floor(np.log10(X.shape[0])))), self.max_batch_size)

        train_dataset, val_dataset = self.generate_datasets(X=X, y=y, params=params, X_val=X_val, y_val=y_val)
        logger.log(15, "Training data for TabularNeuralQuantileModel has: %d examples, %d features "
                       "(%d vector, %d embedding, %d language)" %
                   (train_dataset.num_examples, train_dataset.num_features,
                    len(train_dataset.feature_groups['vector']), len(train_dataset.feature_groups['embed']),
                    len(train_dataset.feature_groups['language'])))

        if num_gpus is not None and num_gpus >= 1:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
                if num_gpus > 1:
                    logger.warning("TabularNeuralQuantileModel not yet configured to use more than 1 GPU."
                                   " 'num_gpus' set to >1, but we will be using only 1 GPU.")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device("cpu")

        self.get_net(train_dataset, params=params)

        if time_limit is not None:
            time_elapsed = time.time() - start_time
            time_limit_orig = time_limit
            time_limit = time_limit - time_elapsed

            # if 60% of time was spent preprocessing, likely not enough time to train model
            if time_limit <= time_limit_orig * 0.4:
                raise TimeLimitExceeded

        # train network
        self.train_net(train_dataset=train_dataset,
                       params=params,
                       val_dataset=val_dataset,
                       initialize=True,
                       setup_trainer=True,
                       time_limit=time_limit,
                       reporter=reporter)
        self.params_post_fit = params

    def get_net(self, train_dataset, params):
        # set network params
        self.set_net_defaults(train_dataset, params)
        self.model = NeuralMultiQuantileRegressor(quantile_levels=self.quantile_levels,
                                                  train_dataset=train_dataset, params=params, device=self.device)
        self.model = self.model.to(self.device)
        if not os.path.exists(self.path):
            os.makedirs(self.path)

    def train_net(self, train_dataset, params, val_dataset=None, initialize=True, setup_trainer=True, time_limit=None, reporter=None):
        start_time = time.time()
        import torch
        logger.log(15, "Training neural network for quantile prediction for up to %s updates..." % params['num_updates'])
        seed_value = params.get('seed_value')
        if seed_value is not None:  # Set seed
            random.seed(seed_value)
            np.random.seed(seed_value)
            torch.manual_seed(seed_value)
        if initialize:
            logging.debug("initializing neural network...")
            self.model.init_params()
            logging.debug("initialized")
        if setup_trainer:
            self.optimizer = self.setup_trainer(params=params)
        train_dataloader = train_dataset.build_loader(self.batch_size, self.num_dataloading_workers, is_test=False)

        best_val_metric = -np.inf  # higher = better
        best_val_update = 0
        val_improve_update = 0  # most recent update where validation-score strictly improved
        num_updates = params['num_updates']
        updates_wo_improve = params['updates_wo_improve']
        if val_dataset is not None:
            y_val = val_dataset.get_labels()
        else:
            y_val = None

        if self.verbosity <= 1:
            verbose_eval = False
        else:
            verbose_eval = True

        net_filename = self.path + self.temp_file_name
        if num_updates == 0:
            # use dummy training loop that stops immediately
            # useful for using NN just for data preprocessing / debugging
            logger.log(20, "Not training Neural Net since num_updates == 0.  Neural network architecture is:")

            # for each batch
            for batch_idx, data_batch in enumerate(train_dataloader):
                loss = self.model.compute_loss(data_batch, weight=1.0, margin=params['gamma'])
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                if batch_idx > 0:
                    break

            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            torch.save(self.model, net_filename)
            logger.log(15, "untrained Quantile Neural Network saved to file")
            return

        start_fit_time = time.time()
        if time_limit is not None:
            time_limit = time_limit - (start_fit_time - start_time)

        # start training Loop:
        logger.log(15, "Start training Qunatile Neural network")
        total_updates = 0
        do_update = True
        while do_update:
            total_train_loss = 0.0
            total_train_size = 0.0
            for batch_idx, data_batch in enumerate(train_dataloader):
                # forward
                weight = (np.cos(min((total_updates / float(updates_wo_improve)), 1.0) * np.pi) + 1) * 0.5
                loss = self.model.compute_loss(data_batch, weight=weight, margin=params['gamma'])
                total_train_loss += loss.item()
                total_train_size += 1

                # update
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_updates += 1

                # validation
                if total_updates % 100 == 0 and val_dataset is not None:
                    # compute validation score
                    val_metric = self.score(X=val_dataset, y=y_val, metric=self.stopping_metric)
                    if np.isnan(val_metric):
                        if total_updates == 1:
                            raise RuntimeError("NaNs encountered in TabularNeuralQuantileModel training. "
                                               "Features/labels may be improperly formatted, "
                                               "or NN weights may have diverged.")
                        else:
                            logger.warning("Warning: NaNs encountered in TabularNeuralQuantileModel training. "
                                           "Reverting model to last checkpoint without NaNs.")
                            break

                    # update best validation
                    if (val_metric >= best_val_metric) or (total_updates == 1):
                        if val_metric > best_val_metric:
                            val_improve_update = total_updates
                        best_val_metric = val_metric
                        best_val_update = total_updates
                        os.makedirs(os.path.dirname(self.path), exist_ok=True)
                        torch.save(self.model, net_filename)
                    if verbose_eval:
                        logger.log(15, "Update %s.  Train loss: %s, Val %s: %s" %
                                   (total_updates, total_train_loss/total_train_size, self.stopping_metric.name, val_metric))

                    if reporter is not None:
                        reporter(epoch=total_updates,
                                 validation_performance=val_metric,  # Higher val_metric = better
                                 train_loss=total_train_loss/total_train_size,
                                 eval_metric=self.eval_metric.name,
                                 greater_is_better=self.eval_metric.greater_is_better)

                    # no improvement
                    if total_updates - val_improve_update > updates_wo_improve:
                        do_update = False
                        break

                elif total_updates % 100 == 0:
                    best_val_update = total_updates
                    if verbose_eval:
                        logger.log(15, "Update %s.  Train loss: %s" % (total_updates, total_train_loss/total_train_size))

                # time limit
                if time_limit is not None:
                    time_elapsed = time.time() - start_fit_time
                    if time_limit < time_elapsed:
                        logger.log(15, f"\tRan out of time, stopping training early. (Stopping on updates {total_updates})")
                        do_update = False
                        break

                # max updates
                if total_updates == num_updates:
                    logger.log(15, f"\tReached the max number of updates. ({num_updates})")
                    do_update = False
                    break

        # revert back to best model
        if val_dataset is not None:
            try:
                self.model = torch.load(net_filename)
                os.remove(net_filename)
            except FileNotFoundError:
                pass

            # evaluate one final time
            final_val_metric = self.score(X=val_dataset, y=y_val, metric=self.stopping_metric)
            if np.isnan(final_val_metric):
                final_val_metric = -np.inf
            logger.log(15, "Best model found in updates %d. Val %s: %s" %
                       (best_val_update, self.stopping_metric.name, final_val_metric))
        else:
            logger.log(15, "Best model found in updates %d" % best_val_update)
        self.params_trained['num_updates'] = best_val_update
        return

    def _predict_proba(self, X, **kwargs):
        """ To align predict with abstract_model API.
            Preprocess here only refers to feature processing steps done by all AbstractModel objects,
            not tabularNN-specific preprocessing steps.
            If X is not DataFrame but instead TabularNNDataset object, we can still produce predictions,
            but cannot use preprocess in this case (needs to be already processed).
        """
        if isinstance(X, TabularPyTorchDataset):
            return self._predict_tabular_data(new_data=X, process=False)
        elif isinstance(X, pd.DataFrame):
            X = self.preprocess(X, **kwargs)
            return self._predict_tabular_data(new_data=X, process=True)
        else:
            raise ValueError("X must be of type pd.DataFrame or TabularPyTorchDataset, not type: %s" % type(X))

    def _predict_tabular_data(self, new_data, process=True, predict_proba=True):
        if process:
            new_data = self.process_test_data(new_data, None)
        if not isinstance(new_data, TabularPyTorchDataset):
            raise ValueError("new_data must of of type TabularNNDataset if process=False")
        val_dataloader = new_data.build_loader(self.max_batch_size, self.num_dataloading_workers, is_test=True)
        preds_dataset = []
        for batch_idx, data_batch in enumerate(val_dataloader):
            preds_batch = self.model.predict(data_batch)
            preds_dataset.append(preds_batch)
        preds_dataset = np.concatenate(preds_dataset, 0)
        return preds_dataset

    def generate_datasets(self, X, y, params, X_val=None, y_val=None):
        impute_strategy = params['proc.impute_strategy']
        max_category_levels = params['proc.max_category_levels']
        skew_threshold = params['proc.skew_threshold']
        embed_min_categories = params['proc.embed_min_categories']
        use_ngram_features = params['use_ngram_features']

        if isinstance(X, TabularPyTorchDataset):
            train_dataset = X
        else:
            X = self.preprocess(X)
            if self.features is None:
                self.features = list(X.columns)
            train_dataset = self.process_train_data(df=X, labels=y,
                                                    impute_strategy=impute_strategy,
                                                    max_category_levels=max_category_levels,
                                                    skew_threshold=skew_threshold,
                                                    embed_min_categories=embed_min_categories,
                                                    use_ngram_features=use_ngram_features)
        if X_val is not None:
            if isinstance(X_val, TabularPyTorchDataset):
                val_dataset = X_val
            else:
                X_val = self.preprocess(X_val)
                val_dataset = self.process_test_data(df=X_val, labels=y_val)
        else:
            val_dataset = None
        return train_dataset, val_dataset

    def process_test_data(self, df, labels=None, **kwargs):
        """ Process train or test DataFrame into a form fit for neural network models.
            Args:
                df (pd.DataFrame): Data to be processed (X)
                labels (pd.Series): labels to be processed (y)
            Returns:
                Dataset object
        """
        # sklearn processing n_quantiles warning
        warnings.filterwarnings("ignore", module='sklearn.preprocessing')
        if labels is not None and len(labels) != len(df):
            raise ValueError("Number of examples in Dataframe does not match number of labels")
        if (self.processor is None or self._types_of_features is None
           or self.feature_arraycol_map is None or self.feature_type_map is None):
            raise ValueError("Need to process training data before test data")
        if self.features_to_drop:
            drop_cols = [col for col in df.columns if col in self.features_to_drop]
            if drop_cols:
                df = df.drop(columns=drop_cols)

        # self.feature_arraycol_map, self.feature_type_map have been previously set while processing training data.
        df = self.processor.transform(df)
        return TabularPyTorchDataset(df, self.feature_arraycol_map, self.feature_type_map, labels)

    def process_train_data(self, df, impute_strategy, max_category_levels, skew_threshold,
                           embed_min_categories, use_ngram_features, labels, **kwargs):
        # sklearn processing n_quantiles warning
        warnings.filterwarnings("ignore", module='sklearn.preprocessing')
        if set(df.columns) != set(self.features):
            raise ValueError("Column names in provided Dataframe do not match self.features")
        if labels is None:
            raise ValueError("Attempting process training data without labels")
        if len(labels) != len(df):
            raise ValueError("Number of examples in Dataframe does not match number of labels")

        # dict with keys: : 'continuous', 'skewed', 'onehot', 'embed', 'language', values = column-names of df
        self._types_of_features, df = self._get_types_of_features(df, skew_threshold=skew_threshold,
                                                                  embed_min_categories=embed_min_categories,
                                                                  use_ngram_features=use_ngram_features)
        logger.log(15, "AutoGluon Qunatile Neural Network (pytorch) infers features are of the following types:")
        logger.log(15, json.dumps(self._types_of_features, indent=4))
        logger.log(15, "\n")
        self.processor = self._create_preprocessor(impute_strategy=impute_strategy,
                                                   max_category_levels=max_category_levels)
        df = self.processor.fit_transform(df)

        # OrderedDict of feature-name -> list of column-indices in df corresponding to this feature
        self.feature_arraycol_map = self._get_feature_arraycol_map(max_category_levels=max_category_levels)

        # should match number of columns in processed array
        num_array_cols = np.sum([len(self.feature_arraycol_map[key]) for key in self.feature_arraycol_map])
        if num_array_cols != df.shape[1]:
            raise ValueError("Error during one-hot encoding data processing for neural network."
                             " Number of columns in df array does not match feature_arraycol_map.")

        # OrderedDict of feature-name -> feature_type string (options: 'vector', 'embed', 'language')
        self.feature_type_map = self._get_feature_type_map()
        return TabularPyTorchDataset(df, self.feature_arraycol_map, self.feature_type_map, labels)

    def setup_trainer(self, params, **kwargs):
        """
        Set up optimizer needed for training.
        Network must first be initialized before this.
        """
        import torch
        if params['optimizer'] == 'sgd':
            optimizer = torch.optim.SGD(params=self.model.parameters(),
                                        lr=params['learning_rate'],
                                        weight_decay=params['weight_decay'])
        elif params['optimizer'] == 'adam':
            optimizer = torch.optim.Adam(params=self.model.parameters(),
                                         lr=params['learning_rate'],
                                         weight_decay=params['weight_decay'])
        else:
            raise ValueError("Unknown optimizer specified: %s" % params['optimizer'])
        return optimizer

    def save(self, path: str = None, verbose=True) -> str:
        if self.model is not None:
            self._architecture_desc = self.model.architecture_desc
        temp_model = self.model
        self.model = None
        path_final = super().save(path=path, verbose=verbose)
        self.model = temp_model
        self._architecture_desc = None

        # Export model
        if self.model is not None:
            import torch
            params_filepath = path_final + self.params_file_name
            # TODO: Don't use os.makedirs here, have save_parameters function in tabular_nn_model that checks if local path or S3 path
            os.makedirs(os.path.dirname(path_final), exist_ok=True)
            torch.save(self.model, params_filepath)
        return path_final

    @classmethod
    def load(cls, path: str, reset_paths=True, verbose=True):
        model: TabularNeuralQuantileModel = AbstractNeuralNetworkModel.load(path=path, reset_paths=reset_paths, verbose=verbose)
        if model._architecture_desc is not None:
            import torch
            # recreate network from architecture description
            model.model = NeuralMultiQuantileRegressor(quantile_levels=model.quantile_levels,
                                                       architecture_desc=model._architecture_desc,
                                                       device=model.device)
            model._architecture_desc = None
            model.model = torch.load(model.path + model.params_file_name)
        return model

    def _hyperparameter_tune(self, X, y, X_val, y_val, scheduler_options, **kwargs):
        """ Performs HPO and sets self.params to best hyperparameter values """
        try_import_torch()
        time_start = time.time()
        self.verbosity = kwargs.get('verbosity', 2)
        logger.log(15, "Beginning hyperparameter tuning for Neural Network...")

        # changes non-specified default hyperparams from fixed values to search-spaces.
        self._set_default_searchspace()
        if self.feature_metadata is None:
            raise ValueError("Trainer class must set feature_metadata for this model")
        scheduler_cls, scheduler_params = scheduler_options  # Unpack tuple
        if scheduler_cls is None or scheduler_params is None:
            raise ValueError("scheduler_cls and scheduler_params cannot be None for hyperparameter tuning")
        num_cpus = scheduler_params['resource']['num_cpus']

        params_copy = self.params.copy()

        self.num_dataloading_workers = max(1, int(num_cpus/2.0))
        self.max_batch_size = params_copy['max_batch_size']
        self.batch_size = min(int(2 ** (3 + np.floor(np.log10(X.shape[0])))), self.max_batch_size)
        train_dataset, val_dataset = self.generate_datasets(X=X, y=y, params=params_copy, X_val=X_val, y_val=y_val)
        train_path = self.path + "train"
        val_path = self.path + "validation"
        train_dataset.save(file_prefix=train_path)
        val_dataset.save(file_prefix=val_path)

        if not np.any([isinstance(params_copy[hyperparam], Space) for hyperparam in params_copy]):
            logger.warning("Warning: Attempting to do hyperparameter optimization without any search space (all hyperparameters are already fixed values)")
        else:
            logger.log(15, "Hyperparameter search space for Neural Network: ")
            for hyperparam in params_copy:
                if isinstance(params_copy[hyperparam], Space):
                    logger.log(15, str(hyperparam)+ ":   "+str(params_copy[hyperparam]))

        util_args = dict(
            train_path=train_path,
            val_path=val_path,
            model=self,
            time_start=time_start,
            time_limit=scheduler_params['time_out'],
            fit_kwargs=scheduler_params['resource'],
        )
        tabular_pytorch_trial.register_args(util_args=util_args, **params_copy)
        scheduler = scheduler_cls(tabular_pytorch_trial, **scheduler_params)
        if ('dist_ip_addrs' in scheduler_params) and (len(scheduler_params['dist_ip_addrs']) > 0):
            # TODO: Ensure proper working directory setup on remote machines
            # This is multi-machine setting, so need to copy dataset to workers:
            logger.log(15, "Uploading preprocessed data to remote workers...")
            scheduler.upload_files([
                train_path + TabularPyTorchDataset.DATAOBJ_SUFFIX,
                val_path + TabularPyTorchDataset.DATAOBJ_SUFFIX,
            ])  # TODO: currently does not work.
            logger.log(15, "uploaded")

        scheduler.run()
        scheduler.join_jobs()

        return self._get_hpo_results(scheduler=scheduler, scheduler_params=scheduler_params, time_start=time_start)


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


@args()
def tabular_pytorch_trial(args, reporter):
    """ Training and evaluation function used during a single trial of HPO """
    try:
        model, args, util_args = model_trial.prepare_inputs(args=args)

        train_dataset = TabularPyTorchDataset.load(util_args.train_path)
        val_dataset = TabularPyTorchDataset.load(util_args.val_path)
        y_val = val_dataset.get_labels()

        fit_model_args = dict(X=train_dataset, y=None, X_val=val_dataset, **util_args.get('fit_kwargs', dict()))
        predict_proba_args = dict(X=val_dataset)
        model_trial.fit_and_save_model(model=model, params=args, fit_args=fit_model_args, predict_proba_args=predict_proba_args, y_val=y_val,
                                       time_start=util_args.time_start, time_limit=util_args.get('time_limit', None), reporter=reporter)
    except Exception as e:
        if not isinstance(e, TimeLimitExceeded):
            logger.exception(e, exc_info=True)
        reporter.terminate()
