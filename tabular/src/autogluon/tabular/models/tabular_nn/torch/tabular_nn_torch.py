import json
import logging
import os
import psutil
import random
import time
import warnings
import numpy as np
import pandas as pd

from autogluon.common.features.types import R_OBJECT, S_TEXT_NGRAM, S_TEXT_AS_CATEGORY
from autogluon.core.constants import BINARY, MULTICLASS, REGRESSION, SOFTCLASS, QUANTILE
from autogluon.core.utils import try_import_torch
from autogluon.core.utils.exceptions import TimeLimitExceeded
from autogluon.core.models.abstract.abstract_nn_model import AbstractNeuralNetworkModel

from ..hyperparameters.parameters import get_default_param
from ..hyperparameters.searchspaces import get_default_searchspace
from ..utils.data_preprocessor import create_preprocessor, get_feature_arraycol_map, get_feature_type_map
from ..utils.nn_architecture_utils import infer_y_range

logger = logging.getLogger(__name__)


# TODO: QuantileTransformer in pipelines accounts for majority of online inference time
class TabularNeuralNetTorchModel(AbstractNeuralNetworkModel):
    """
    PyTorch neural network models for classification/regression with tabular data.
    """

    # Constants used throughout this class:
    unique_category_str = '!missing!'  # string used to represent missing values and unknown categories for categorical features.
    params_file_name = 'net.params'  # Stores parameters of final network
    temp_file_name = 'temp_net.params'  # Stores temporary network parameters (eg. during the course of training)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.feature_arraycol_map = None
        self.feature_type_map = None
        self.features_to_drop = []  # may change between different bagging folds. TODO: consider just removing these from self._features_internal
        self.processor = None  # data processor
        self.num_dataloading_workers = None
        self._architecture_desc = None
        self.optimizer = None
        self.device = None
        self.max_batch_size = None

    def _set_default_params(self):
        """ Specifies hyperparameter values to use by default """
        default_params = get_default_param(problem_type=self.problem_type, framework='pytorch')
        for param, val in default_params.items():
            self._set_default_param_value(param, val)

    def _get_default_auxiliary_params(self) -> dict:
        default_auxiliary_params = super()._get_default_auxiliary_params()
        extra_auxiliary_params = dict(
            ignored_type_group_raw=[R_OBJECT],
            ignored_type_group_special=[S_TEXT_NGRAM, S_TEXT_AS_CATEGORY],
        )
        default_auxiliary_params.update(extra_auxiliary_params)
        return default_auxiliary_params

    def _get_default_searchspace(self):
        return get_default_searchspace(problem_type=self.problem_type, framework='pytorch')

    def _get_num_net_outputs(self):
        if self.problem_type in [MULTICLASS, SOFTCLASS]:
            return self.num_classes
        elif self.problem_type == BINARY:
            return 2
        elif self.problem_type == REGRESSION:
            return 1
        elif self.problem_type == QUANTILE:
            return len(self.quantile_levels)
        else:
            raise ValueError(f'Unknown problem_type: {self.problem_type}')

    def _get_device(self, num_gpus):
        import torch
        if num_gpus is not None and num_gpus >= 1:
            if torch.cuda.is_available():
                device = torch.device("cuda")
                logger.log(15, "Training on GPU")
                if num_gpus > 1:
                    logger.warning(f"{self.__class__.__name__} not yet able to use more than 1 GPU. 'num_gpus' is set to >1, but we will be using only 1 GPU.")
            else:
                device = torch.device("cpu")
                logger.log(15, "Training on CPU")
        else:
            device = torch.device("cpu")
            logger.log(15, "Training on CPU")
        return device

    def _set_net_defaults(self, train_dataset, params):
        params = params.copy()
        y_range_extend = params.pop('y_range_extend', None)
        """ Sets dataset-adaptive default values to use for our neural network """
        if self.problem_type in [REGRESSION, QUANTILE]:
            if params['y_range'] is None:
                params['y_range'] = infer_y_range(y_vals=train_dataset.data_list[train_dataset.label_index], y_range_extend=y_range_extend)
        return params

    def _get_default_loss_function(self):
        import torch
        if self.problem_type == REGRESSION:
            return torch.nn.L1Loss()  # or torch.nn.MSELoss()
        elif self.problem_type in [BINARY, MULTICLASS]:
            return torch.nn.CrossEntropyLoss()
        elif self.problem_type == SOFTCLASS:
            return torch.nn.KLDivLoss()  # compares log-probability prediction vs probability target.

    @staticmethod
    def _prepare_params(params):
        params = params.copy()

        processor_param_keys = {'proc.embed_min_categories', 'proc.impute_strategy', 'proc.max_category_levels', 'proc.skew_threshold', 'use_ngram_features'}
        processor_kwargs = {k: v for k, v in params.items() if k in processor_param_keys}
        for key in processor_param_keys:
            params.pop(key, None)

        optimizer_param_keys = {'optimizer', 'learning_rate', 'weight_decay'}
        optimizer_kwargs = {k: v for k, v in params.items() if k in optimizer_param_keys}
        for key in optimizer_param_keys:
            params.pop(key, None)

        fit_param_keys = {'num_epochs', 'epochs_wo_improve'}
        fit_kwargs = {k: v for k, v in params.items() if k in fit_param_keys}
        for key in fit_param_keys:
            params.pop(key, None)

        loss_param_keys = {'loss_function', 'gamma'}
        loss_kwargs = {k: v for k, v in params.items() if k in loss_param_keys}
        for key in loss_param_keys:
            params.pop(key, None)

        return processor_kwargs, optimizer_kwargs, fit_kwargs, loss_kwargs, params

    def _fit(self, X, y, X_val=None, y_val=None,
             time_limit=None, sample_weight=None, num_cpus=1, num_gpus=0, reporter=None, verbosity=2, **kwargs):
        try_import_torch()
        import torch
        torch.set_num_threads(num_cpus)
        from .tabular_torch_dataset import TabularTorchDataset

        start_time = time.time()

        params = self._get_model_params()

        processor_kwargs, optimizer_kwargs, fit_kwargs, loss_kwargs, params = self._prepare_params(params=params)

        seed_value = params.pop('seed_value', 0)
        if seed_value is not None:  # Set seeds
            random.seed(seed_value)
            np.random.seed(seed_value)
            torch.manual_seed(seed_value)

        if sample_weight is not None:  # TODO: support
            logger.log(15, f"sample_weight not yet supported for {self.__class__.__name__},"
                           " this model will ignore them in training.")

        if num_cpus is not None:
            self.num_dataloading_workers = max(1, int(num_cpus/2.0))
        else:
            self.num_dataloading_workers = 1
        if self.num_dataloading_workers == 1:
            self.num_dataloading_workers = 0  # TODO: verify 0 is typically faster and uses less memory than 1 in pytorch
        self.num_dataloading_workers = 0  # TODO: >0 crashes on MacOS
        self.max_batch_size = params.pop('max_batch_size', 512)
        batch_size = params.pop('batch_size', None)
        if batch_size is None:
            if isinstance(X, TabularTorchDataset):
                batch_size = min(int(2 ** (3 + np.floor(np.log10(len(X))))), self.max_batch_size)
            else:
                batch_size = min(int(2 ** (3 + np.floor(np.log10(X.shape[0])))), self.max_batch_size)

        train_dataset, val_dataset = self._generate_datasets(X=X, y=y, params=processor_kwargs, X_val=X_val, y_val=y_val)
        logger.log(15, f"Training data for {self.__class__.__name__} has: "
                       f"{train_dataset.num_examples} examples, {train_dataset.num_features} features "
                       f"({len(train_dataset.feature_groups['vector'])} vector, {len(train_dataset.feature_groups['embed'])} embedding)")

        self.device = self._get_device(num_gpus=num_gpus)

        self._get_net(train_dataset, params=params)
        self.optimizer = self._init_optimizer(**optimizer_kwargs)

        if time_limit is not None:
            time_elapsed = time.time() - start_time
            time_limit_orig = time_limit
            time_limit = time_limit - time_elapsed

            # if 60% of time was spent preprocessing, likely not enough time to train model
            if time_limit <= time_limit_orig * 0.4:
                raise TimeLimitExceeded

        # train network
        self._train_net(train_dataset=train_dataset,
                        loss_kwargs=loss_kwargs,
                        batch_size=batch_size,
                        val_dataset=val_dataset,
                        time_limit=time_limit,
                        reporter=reporter,
                        verbosity=verbosity,
                        **fit_kwargs)

    def _get_net(self, train_dataset, params):
        from .torch_network_modules import EmbedNet

        # set network params
        params = self._set_net_defaults(train_dataset, params)
        self.model = EmbedNet(problem_type=self.problem_type, num_net_outputs=self._get_num_net_outputs(), quantile_levels=self.quantile_levels,
                              train_dataset=train_dataset, device=self.device, **params)
        self.model = self.model.to(self.device)
        if not os.path.exists(self.path):
            os.makedirs(self.path)

    def _train_net(self,
                   train_dataset,
                   loss_kwargs,
                   batch_size,
                   num_epochs,
                   epochs_wo_improve,
                   val_dataset=None,
                   time_limit=None,
                   reporter=None,
                   verbosity=2):
        import torch
        start_time = time.time()
        logging.debug("initializing neural network...")
        self.model.init_params()
        logging.debug("initialized")
        train_dataloader = train_dataset.build_loader(batch_size, self.num_dataloading_workers, is_test=False)

        if isinstance(loss_kwargs.get('loss_function', 'auto'), str) and loss_kwargs.get('loss_function', 'auto') == 'auto':
            loss_kwargs['loss_function'] = self._get_default_loss_function()

        if val_dataset is not None:
            y_val = val_dataset.get_labels()
            if y_val.ndim == 2 and y_val.shape[1] == 1:
                y_val = y_val.flatten()
        else:
            y_val = None

        if verbosity <= 1:
            verbose_eval = False
        else:
            verbose_eval = True

        logger.log(15, "Neural network architecture:")
        logger.log(15, str(self.model))

        net_filename = self.path + self.temp_file_name
        if num_epochs == 0:
            # use dummy training loop that stops immediately
            # useful for using NN just for data preprocessing / debugging
            logger.log(20, "Not training Tabular Neural Network since num_updates == 0")

            # for each batch
            for batch_idx, data_batch in enumerate(train_dataloader):
                if batch_idx > 0:
                    break
                loss = self.model.compute_loss(data_batch, **loss_kwargs)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            torch.save(self.model, net_filename)
            logger.log(15, "Untrained Tabular Neural Network saved to file")
            return

        # start training loop:
        logger.log(15, f"Training tabular neural network for up to {num_epochs} epochs...")
        total_updates = 0
        num_updates_per_epoch = len(train_dataloader)
        update_to_check_time = min(10, max(1, int(num_updates_per_epoch/10)))
        do_update = True
        epoch = 0
        best_epoch = 0
        best_val_metric = -np.inf  # higher = better
        best_val_update = 0
        val_improve_epoch = 0  # most recent epoch where validation-score strictly improved
        start_fit_time = time.time()
        if time_limit is not None:
            time_limit = time_limit - (start_fit_time - start_time)
            if time_limit <= 0:
                raise TimeLimitExceeded
        while do_update:
            time_start_epoch = time.time()
            total_train_loss = 0.0
            total_train_size = 0.0
            for batch_idx, data_batch in enumerate(train_dataloader):
                # forward
                loss = self.model.compute_loss(data_batch, **loss_kwargs)
                total_train_loss += loss.item()
                total_train_size += 1

                # update
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_updates += 1

                # time limit
                if time_limit is not None:
                    time_cur = time.time()
                    update_cur = batch_idx + 1
                    if epoch == 0 and update_cur == update_to_check_time:
                        time_elapsed_epoch = time_cur - time_start_epoch
                        estimated_time = time_elapsed_epoch / update_cur * num_updates_per_epoch
                        if estimated_time > time_limit:
                            logger.log(30, f"\tNot enough time to train first epoch. "
                                           f"(Time Required: {round(estimated_time, 2)}s, Time Left: {round(time_limit, 2)}s)")
                            raise TimeLimitExceeded
                    time_elapsed = time_cur - start_fit_time
                    if time_limit < time_elapsed:
                        logger.log(15, f"\tRan out of time, stopping training early. (Stopped on Update {total_updates} (Epoch {epoch}))")
                        do_update = False
                        break

            if not do_update:
                break

            epoch += 1

            # validation
            if val_dataset is not None:
                # compute validation score
                val_metric = self.score(X=val_dataset, y=y_val, metric=self.stopping_metric)
                if np.isnan(val_metric):
                    if best_epoch == 0:
                        raise RuntimeError(f"NaNs encountered in {self.__class__.__name__} training. "
                                           "Features/labels may be improperly formatted, "
                                           "or NN weights may have diverged.")
                    else:
                        logger.warning(f"Warning: NaNs encountered in {self.__class__.__name__} training. "
                                       "Reverting model to last checkpoint without NaNs.")
                        break

                # update best validation
                if (val_metric >= best_val_metric) or best_epoch == 0:
                    if val_metric > best_val_metric:
                        val_improve_epoch = epoch
                    best_val_metric = val_metric
                    os.makedirs(os.path.dirname(self.path), exist_ok=True)
                    torch.save(self.model, net_filename)
                    best_epoch = epoch
                    best_val_update = total_updates
                if verbose_eval:
                    logger.log(15, f"Epoch {epoch} (Update {total_updates}).\t"
                                   f"Train loss: {round(total_train_loss / total_train_size, 4)}, "
                                   f"Val {self.stopping_metric.name}: {round(val_metric, 4)}, "
                                   f"Best Epoch: {best_epoch}")

                if reporter is not None:
                    reporter(epoch=total_updates,
                             validation_performance=val_metric,  # Higher val_metric = better
                             train_loss=total_train_loss / total_train_size,
                             eval_metric=self.eval_metric.name,
                             greater_is_better=self.eval_metric.greater_is_better)

                # no improvement
                if epoch - val_improve_epoch >= epochs_wo_improve:
                    break

            if epoch >= num_epochs:
                break

            if time_limit is not None:
                time_elapsed = time.time() - start_fit_time
                time_epoch_average = time_elapsed / (epoch+1)
                time_left = time_limit - time_elapsed
                if time_left < time_epoch_average:
                    logger.log(20, f"\tRan out of time, stopping training early. (Stopping on epoch {epoch})")
                    break

        if epoch == 0:
            raise AssertionError('0 epochs trained!')

        # revert back to best model
        if val_dataset is not None:
            logger.log(15, f"Best model found on Epoch {best_epoch} (Update {best_val_update}). Val {self.stopping_metric.name}: {best_val_metric}")
            try:
                self.model = torch.load(net_filename)
                os.remove(net_filename)
            except FileNotFoundError:
                pass
        else:
            logger.log(15, f"Best model found on Epoch {best_epoch} (Update {best_val_update}).")
        self.params_trained['batch_size'] = batch_size
        self.params_trained['num_epochs'] = best_epoch

    def _predict_proba(self, X, **kwargs):
        """ To align predict with abstract_model API.
            Preprocess here only refers to feature processing steps done by all AbstractModel objects,
            not tabularNN-specific preprocessing steps.
            If X is not DataFrame but instead TabularNNDataset object, we can still produce predictions,
            but cannot use preprocess in this case (needs to be already processed).
        """
        from .tabular_torch_dataset import TabularTorchDataset
        if isinstance(X, TabularTorchDataset):
            return self._predict_tabular_data(new_data=X, process=False)
        elif isinstance(X, pd.DataFrame):
            X = self.preprocess(X, **kwargs)
            return self._predict_tabular_data(new_data=X, process=True)
        else:
            raise ValueError("X must be of type pd.DataFrame or TabularTorchDataset, not type: %s" % type(X))

    def _predict_tabular_data(self, new_data, process=True):
        from .tabular_torch_dataset import TabularTorchDataset
        if process:
            new_data = self._process_test_data(new_data)
        if not isinstance(new_data, TabularTorchDataset):
            raise ValueError("new_data must of of type TabularTorchDataset if process=False")
        val_dataloader = new_data.build_loader(self.max_batch_size, self.num_dataloading_workers, is_test=True)
        preds_dataset = []
        for batch_idx, data_batch in enumerate(val_dataloader):
            preds_batch = self.model.predict(data_batch)
            preds_dataset.append(preds_batch)
        preds_dataset = np.concatenate(preds_dataset, 0)
        return preds_dataset

    def _generate_datasets(self, X, y, params, X_val=None, y_val=None):
        from .tabular_torch_dataset import TabularTorchDataset

        impute_strategy = params['proc.impute_strategy']
        max_category_levels = params['proc.max_category_levels']
        skew_threshold = params['proc.skew_threshold']
        embed_min_categories = params['proc.embed_min_categories']
        use_ngram_features = params['use_ngram_features']

        if isinstance(X, TabularTorchDataset):
            train_dataset = X
        else:
            X = self.preprocess(X)
            train_dataset = self._process_train_data(df=X, labels=y,
                                                     impute_strategy=impute_strategy,
                                                     max_category_levels=max_category_levels,
                                                     skew_threshold=skew_threshold,
                                                     embed_min_categories=embed_min_categories,
                                                     use_ngram_features=use_ngram_features)
        if X_val is not None:
            if isinstance(X_val, TabularTorchDataset):
                val_dataset = X_val
            else:
                X_val = self.preprocess(X_val)
                val_dataset = self._process_test_data(df=X_val, labels=y_val)
        else:
            val_dataset = None
        return train_dataset, val_dataset

    def _process_test_data(self, df, labels=None):
        """ Process train or test DataFrame into a form fit for neural network models.
            Args:
                df (pd.DataFrame): Data to be processed (X)
                labels (pd.Series): labels to be processed (y)
            Returns:
                Dataset object
        """
        from .tabular_torch_dataset import TabularTorchDataset

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
        return TabularTorchDataset(df, self.feature_arraycol_map, self.feature_type_map, self.problem_type, labels)

    def _process_train_data(self, df, impute_strategy, max_category_levels, skew_threshold,
                            embed_min_categories, use_ngram_features, labels):
        from .tabular_torch_dataset import TabularTorchDataset

        # sklearn processing n_quantiles warning
        warnings.filterwarnings("ignore", module='sklearn.preprocessing')
        if labels is None:
            raise ValueError("Attempting process training data without labels")
        if len(labels) != len(df):
            raise ValueError("Number of examples in Dataframe does not match number of labels")

        # dict with keys: : 'continuous', 'skewed', 'onehot', 'embed', values = column-names of df
        self._types_of_features, df = self._get_types_of_features(df, skew_threshold=skew_threshold,
                                                                  embed_min_categories=embed_min_categories,
                                                                  use_ngram_features=use_ngram_features)
        logger.log(15, "Tabular Neural Network treats features as the following types:")
        logger.log(15, json.dumps(self._types_of_features, indent=4))
        logger.log(15, "\n")
        if self.processor is not None:
            Warning(f"Attempting to process training data for {self.__class__.__name__}, but previously already did this.")
        self.processor = create_preprocessor(
            impute_strategy=impute_strategy,
            max_category_levels=max_category_levels,
            unique_category_str=self.unique_category_str,
            continuous_features=self._types_of_features['continuous'],
            skewed_features=self._types_of_features['skewed'],
            onehot_features=self._types_of_features['onehot'],
            embed_features=self._types_of_features['embed'],
            bool_features=self._types_of_features['bool']
        )
        df = self.processor.fit_transform(df)
        # OrderedDict of feature-name -> list of column-indices in df corresponding to this feature
        self.feature_arraycol_map = get_feature_arraycol_map(processor=self.processor, max_category_levels=max_category_levels)
        num_array_cols = np.sum([len(self.feature_arraycol_map[key]) for key in self.feature_arraycol_map])  # should match number of columns in processed array
        if num_array_cols != df.shape[1]:
            raise ValueError("Error during one-hot encoding data processing for neural network. "
                             "Number of columns in df array does not match feature_arraycol_map.")

        # OrderedDict of feature-name -> feature_type string (options: 'vector', 'embed')
        self.feature_type_map = get_feature_type_map(feature_arraycol_map=self.feature_arraycol_map, types_of_features=self._types_of_features)
        return TabularTorchDataset(df, self.feature_arraycol_map, self.feature_type_map, self.problem_type, labels)

    def _init_optimizer(self, optimizer, learning_rate, weight_decay):
        """
        Set up optimizer needed for training.
        Network must first be initialized before this.
        """
        import torch

        if optimizer == 'sgd':
            optimizer = torch.optim.SGD(params=self.model.parameters(),
                                        lr=learning_rate,
                                        weight_decay=weight_decay)
        elif optimizer == 'adam':
            optimizer = torch.optim.Adam(params=self.model.parameters(),
                                         lr=learning_rate,
                                         weight_decay=weight_decay)
        else:
            raise ValueError(f"Unknown optimizer specified: {optimizer}")
        return optimizer

    def reduce_memory_size(self, remove_fit=True, requires_save=True, **kwargs):
        super().reduce_memory_size(remove_fit=remove_fit, requires_save=requires_save, **kwargs)
        if remove_fit and requires_save:
            self.optimizer = None

    def _get_default_stopping_metric(self):
        return self.eval_metric

    def _get_default_resources(self):
        # psutil.cpu_count(logical=False) is faster in training than psutil.cpu_count()
        num_cpus = psutil.cpu_count(logical=False)
        num_gpus = 0
        return num_cpus, num_gpus

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
        model: TabularNeuralNetTorchModel = super().load(path=path, reset_paths=reset_paths, verbose=verbose)
        if model._architecture_desc is not None:
            import torch
            from .torch_network_modules import EmbedNet

            # recreate network from architecture description
            model.model = EmbedNet(problem_type=model.problem_type,
                                   num_net_outputs=model._get_num_net_outputs(),
                                   quantile_levels=model.quantile_levels,
                                   architecture_desc=model._architecture_desc,
                                   device=model.device)
            model._architecture_desc = None
            model.model = torch.load(model.path + model.params_file_name)
        return model
