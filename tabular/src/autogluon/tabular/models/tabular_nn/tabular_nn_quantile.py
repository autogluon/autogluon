import json
import logging
import os
import random
import time
import warnings

import numpy as np
import pandas as pd

from autogluon.core.constants import QUANTILE
from autogluon.core.utils import try_import_torch
from autogluon.core.utils.exceptions import TimeLimitExceeded
from autogluon.core.models.abstract.abstract_model import AbstractNeuralNetworkModel
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

    def _fit(self, X, y, X_val=None, y_val=None,
             time_limit=None, sample_weight=None, num_cpus=1, num_gpus=0, reporter=None, **kwargs):
        try_import_torch()
        import torch
        from .tabular_nn_torch import TabularPyTorchDataset

        start_time = time.time()
        self.verbosity = kwargs.get('verbosity', 2)
        if sample_weight is not None:  # TODO: support
            logger.log(15, "sample_weight not yet supported for TabularNeuralQuantileModel,"
                           " this model will ignore them in training.")
        params = self.params.copy()
        params = fixedvals_from_searchspaces(params)
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
        from .tabular_nn_torch import NeuralMultiQuantileRegressor

        # set network params
        self.set_net_defaults(train_dataset, params)
        self.model = NeuralMultiQuantileRegressor(quantile_levels=self.quantile_levels,
                                                  train_dataset=train_dataset, params=params, device=self.device)
        self.model = self.model.to(self.device)
        if not os.path.exists(self.path):
            os.makedirs(self.path)

    def train_net(self, train_dataset, params, val_dataset=None, initialize=True, setup_trainer=True, time_limit=None, reporter=None):
        import torch
        start_time = time.time()
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
        from .tabular_nn_torch import TabularPyTorchDataset
        if isinstance(X, TabularPyTorchDataset):
            return self._predict_tabular_data(new_data=X, process=False)
        elif isinstance(X, pd.DataFrame):
            X = self.preprocess(X, **kwargs)
            return self._predict_tabular_data(new_data=X, process=True)
        else:
            raise ValueError("X must be of type pd.DataFrame or TabularPyTorchDataset, not type: %s" % type(X))

    def _predict_tabular_data(self, new_data, process=True, predict_proba=True):
        from .tabular_nn_torch import TabularPyTorchDataset
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
        from .tabular_nn_torch import TabularPyTorchDataset

        impute_strategy = params['proc.impute_strategy']
        max_category_levels = params['proc.max_category_levels']
        skew_threshold = params['proc.skew_threshold']
        embed_min_categories = params['proc.embed_min_categories']
        use_ngram_features = params['use_ngram_features']

        if isinstance(X, TabularPyTorchDataset):
            train_dataset = X
        else:
            X = self.preprocess(X)
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
        from .tabular_nn_torch import TabularPyTorchDataset

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
        from .tabular_nn_torch import TabularPyTorchDataset

        # sklearn processing n_quantiles warning
        warnings.filterwarnings("ignore", module='sklearn.preprocessing')
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
            from .tabular_nn_torch import NeuralMultiQuantileRegressor

            # recreate network from architecture description
            model.model = NeuralMultiQuantileRegressor(quantile_levels=model.quantile_levels,
                                                       architecture_desc=model._architecture_desc,
                                                       device=model.device)
            model._architecture_desc = None
            model.model = torch.load(model.path + model.params_file_name)
        return model
