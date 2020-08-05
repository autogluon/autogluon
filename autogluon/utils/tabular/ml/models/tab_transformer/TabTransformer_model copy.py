import torch
from torch import nn
import copy
from .transformer_data_encoders import EmbeddingInitializer
from .TabModelBase import TabModelBase

class TabTransformerModel(AbstractModel):
    """
    Transformer model for tabular data
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        """
        TabularNeuralNetModel object.

        Parameters
        ----------
        path (str): file-path to directory where to save files associated with this model
        name (str): name used to refer to this model
        problem_type (str): what type of prediction problem is this model used for
        eval_metric (func): function used to evaluate performance (Note: we assume higher = better)
        hyperparameters (dict): various hyperparameters for neural network and the NN-specific data processing
        features (list): List of predictive features to use, other features are ignored by the model.
        """
        self.types_of_features = None
        self.feature_arraycol_map = None
        self.feature_type_map = None
        self.processor = None # data processor
        self.summary_writer = None
        self.ctx = mx.cpu()
        self.batch_size = None
        self.num_dataloading_workers = None
        self.num_dataloading_workers_inference = 0
        self.params_post_fit = None
        self.num_net_outputs = None
        self._architecture_desc = None
        self.optimizer = None
        self.verbosity = None
        if self.stopping_metric is not None and self.eval_metric == roc_auc and self.stopping_metric == log_loss:
            self.stopping_metric = roc_auc  # NN is overconfident so early stopping with logloss can halt training too quick

        self.eval_metric_name = self.stopping_metric.name
    def init_input(self, input):
        breakpoint()
        cat_feats, cont_feats = input[0], input[1]
        ds_name = input[2][0] if len(input) == 3 else self.dataset_name

        cat_feats = [init(cat_feats[:, i]) for i, init in enumerate(self.cat_initializers[ds_name].values())]
        if isinstance(cont_feats, torch.Tensor) and self.n_cont_embeddings:
            cont_feats = self.cont_norm(cont_feats)
            cont_feats = self.cont_initializer(cont_feats)
            cont_feats = self.cont_init_norm(cont_feats)
            cont_feats = list(cont_feats.split(self.hidden_dim, dim=1)) if self.n_cont_embeddings > 0 else []
        else:
            cont_feats = []
        if self.readout == 'readout_emb':
            readout_emb = self.readout_emb.expand_as(cat_feats[0] if cat_feats else cont_feats[0])
            feat_embs = torch.stack([readout_emb] + cat_feats + cont_feats,
                                    dim=0)  # (n_feat_embeddings + 1) x batch x hidden_dim
        else:
            feat_embs = torch.stack(cat_feats + cont_feats, dim=0)  # n_feat_embeddings x batch x hidden_dim
        return feat_embs

    def run_tfmr(self, feat_embs):
        orig_feat_embs = feat_embs
        all_feat_embs = [feat_embs]
        for layer in self.tfmr_layers:
            feat_embs = layer(feat_embs)
            all_feat_embs.append(feat_embs)
            if self.orig_emb_resid:
                feat_embs = feat_embs + orig_feat_embs

        if self.readout == 'readout_emb':
            out = self.fc_out(feat_embs[0])
        elif self.readout == 'mean':
            out = torch.mean(feat_embs, dim=0)
            out = self.fc_out(out)
        elif self.readout == 'concat_pool':
            all_feat_embs = torch.cat(all_feat_embs, dim=0)

            max = all_feat_embs.max(dim=0).values
            mean = all_feat_embs.mean(dim=0)
            last_layer = feat_embs.transpose(0, 1).reshape(feat_embs.shape[1], -1)
            out = torch.cat((last_layer, max, mean), dim=1)
            out = self.fc_out(out)
        elif self.readout == 'concat_pool_all':
            feat_embs_all_layers = []
            for each_feat_embs in [all_feat_embs[0], all_feat_embs[-1]]:
                feat_embs_all_layers.append(each_feat_embs.transpose(0, 1).reshape(each_feat_embs.shape[1], -1))

            all_feat_embs = torch.cat(all_feat_embs, dim=0)
            max = all_feat_embs.max(dim=0).values
            mean = all_feat_embs.mean(dim=0)

            feat_embs_all_layers.append(max)
            feat_embs_all_layers.append(mean)
            out = torch.cat(feat_embs_all_layers, dim=1)
            out = self.fc_out(out)
        elif self.readout == 'concat_pool_add':
            orig_feat_embs_cp = copy.deepcopy(orig_feat_embs.detach())
            #ce_dim = orig_feat_embs_cp.shape[-1]//8
            #orig_feat_embs_cp[:, :, ce_dim:] = 0

            last_layer = feat_embs.transpose(0, 1).reshape(feat_embs.shape[1], -1)
            last_layer += orig_feat_embs_cp.transpose(0, 1).reshape(orig_feat_embs_cp.shape[1], -1)

            all_feat_embs = torch.cat(all_feat_embs, dim=0)
            max = all_feat_embs.max(dim=0).values
            mean = all_feat_embs.mean(dim=0)

            out = torch.cat([last_layer, max, mean], dim=1)
            out = self.fc_out(out)


        elif self.readout == 'all_feat_embs':
            out = feat_embs
        if self.act_on_output:
            out = self.get_act()(out)
        return out

    def forward(self, input):
        """
        Returns logits for output classes
        """
        feat_embs = self.init_input(input)
        out = self.run_tfmr(feat_embs)
        return out


    def get_net(self, train_dataset, params):
        pass

    def train_net():
        pass


    def _fit(self, X_train, y_train, X_val=None, y_val=None, time_limit=None, reporter=None, **kwargs):
        """ X_train (pd.DataFrame): training data features (not necessarily preprocessed yet)
            X_val (pd.DataFrame): test data features (should have same column names as Xtrain)
            y_train (pd.Series):
            y_val (pd.Series): are pandas Series
            kwargs: Can specify amount of compute resources to utilize (num_cpus, num_gpus).
        """
        start_time = time.time()
        params = self.params.copy()
        self.verbosity = kwargs.get('verbosity', 2)
        params = fixedvals_from_searchspaces(params)
        if self.feature_types_metadata is None:
            raise ValueError("Trainer class must set feature_types_metadata for this model")
        # print('features: ', self.features)
        if 'num_cpus' in kwargs:
            self.num_dataloading_workers = max(1, int(kwargs['num_cpus']/2.0))
        else:
            self.num_dataloading_workers = 1
        if self.num_dataloading_workers == 1:
            self.num_dataloading_workers = 0  # 0 is always faster and uses less memory than 1
        self.batch_size = params['batch_size']
        train_dataset, val_dataset = self.generate_datasets(X_train=X_train, y_train=y_train, params=params, X_val=X_val, y_val=y_val)
        logger.log(15, "Training data for neural network has: %d examples, %d features (%d vector, %d embedding, %d language)" %
              (train_dataset.num_examples, train_dataset.num_features,
               len(train_dataset.feature_groups['vector']), len(train_dataset.feature_groups['embed']),
               len(train_dataset.feature_groups['language']) ))
        # self._save_preprocessor() # TODO: should save these things for hyperparam tunning. Need one HP tuner for network-specific HPs, another for preprocessing HPs.

        if 'num_gpus' in kwargs and kwargs['num_gpus'] >= 1:  # Currently cannot use >1 GPU
            self.ctx = mx.gpu()  # Currently cannot use more than 1 GPU
        else:
            self.ctx = mx.cpu()
        self.get_net(train_dataset, params=params)

        if time_limit:
            time_elapsed = time.time() - start_time
            time_limit = time_limit - time_elapsed

        self.train_net(train_dataset=train_dataset, params=params, val_dataset=val_dataset, initialize=True, setup_trainer=True, time_limit=time_limit, reporter=reporter)
        self.params_post_fit = params

    def generate_datasets(self, X_train, y_train, params, X_val=None, y_val=None):
        """ Convert potentially un-preprocesses train and validation data into TabularDatasets
        Args:
            df (pd.DataFrame): Data to be processed (X)
            labels (pd.Series): labels to be processed (y)
            test (bool): Is this test data where each datapoint should be processed separately using predetermined preprocessing steps.
                         Otherwise preprocessor uses all data to determine propreties like best scaling factors, number of categories, etc.
        Returns:
            TabTransformerDataset object
        """

        impute_strategy = params['proc.impute_strategy']
        max_category_levels = params['proc.max_category_levels']
        skew_threshold = params['proc.skew_threshold']
        embed_min_categories = params['proc.embed_min_categories']
        use_ngram_features = params['use_ngram_features']

        if isinstance(X_train, TabularNNDataset):
            train_dataset = X_train
        else:
            X_train = self.preprocess(X_train)
            if self.features is None:
                self.features = list(X_train.columns)
            train_dataset = self.process_train_data(
                df=X_train, labels=y_train, batch_size=self.batch_size, num_dataloading_workers=self.num_dataloading_workers,
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
        warnings.filterwarnings("ignore", module='sklearn.preprocessing') # sklearn processing n_quantiles warning
        if set(df.columns) != set(self.features):
            raise ValueError("Column names in provided Dataframe do not match self.features")
        if labels is not None and len(labels) != len(df):
            raise ValueError("Number of examples in Dataframe does not match number of labels")
        if (self.processor is None or self.types_of_features is None
           or self.feature_arraycol_map is None or self.feature_type_map is None):
            raise ValueError("Need to process training data before test data")
        df = self.processor.transform(df) # 2D numpy array. self.feature_arraycol_map, self.feature_type_map have been previously set while processing training data.
        return TabularNNDataset(df, self.feature_arraycol_map, self.feature_type_map,
                                batch_size=batch_size, num_dataloading_workers=num_dataloading_workers,
                                problem_type=self.problem_type, labels=labels, is_test=True)

    def process_train_data(self, df, batch_size, num_dataloading_workers, impute_strategy, max_category_levels, skew_threshold, embed_min_categories, use_ngram_features, labels):
        """ Preprocess training data and create self.processor object that can be used to process future data.
            This method should only be used once per TabularNeuralNetModel object, otherwise will produce Warning.

        # TODO no label processing for now
        # TODO: language features are ignored for now
        # TODO: how to add new features such as time features and remember to do the same for test data?
        # TODO: no filtering of data-frame columns based on statistics, e.g. categorical columns with all unique variables or zero-variance features.
                This should be done in default_learner class for all models not just TabularNeuralNetModel...
        """
        warnings.filterwarnings("ignore", module='sklearn.preprocessing')  # sklearn processing n_quantiles warning
        if set(df.columns) != set(self.features):
            raise ValueError("Column names in provided Dataframe do not match self.features")
        if labels is None:
            raise ValueError("Attempting process training data without labels")
        if len(labels) != len(df):
            raise ValueError("Number of examples in Dataframe does not match number of labels")

        self.types_of_features = self._get_types_of_features(df, skew_threshold=skew_threshold, embed_min_categories=embed_min_categories, use_ngram_features=use_ngram_features) # dict with keys: : 'continuous', 'skewed', 'onehot', 'embed', 'language', values = column-names of df
        df = df[self.features]
        logger.log(15, "AutoGluon Neural Network infers features are of the following types:")
        logger.log(15, json.dumps(self.types_of_features, indent=4))
        logger.log(15, "\n")
        self.processor = self._create_preprocessor(impute_strategy=impute_strategy, max_category_levels=max_category_levels)
        df = self.processor.fit_transform(df) # 2D numpy array
        self.feature_arraycol_map = self._get_feature_arraycol_map(max_category_levels=max_category_levels) # OrderedDict of feature-name -> list of column-indices in df corresponding to this feature
        num_array_cols = np.sum([len(self.feature_arraycol_map[key]) for key in self.feature_arraycol_map]) # should match number of columns in processed array
        # print("self.feature_arraycol_map", self.feature_arraycol_map)
        # print("num_array_cols", num_array_cols)
        # print("df.shape",df.shape)
        if num_array_cols != df.shape[1]:
            raise ValueError("Error during one-hot encoding data processing for neural network. Number of columns in df array does not match feature_arraycol_map.")

        # print(self.feature_arraycol_map)
        self.feature_type_map = self._get_feature_type_map() # OrderedDict of feature-name -> feature_type string (options: 'vector', 'embed', 'language')
        # print(self.feature_type_map)
        return TabTransformerDataset(df, self.feature_arraycol_map, self.feature_type_map,
                                batch_size=batch_size, num_dataloading_workers=num_dataloading_workers,
                                problem_type=self.problem_type, labels=labels, is_test=False)

