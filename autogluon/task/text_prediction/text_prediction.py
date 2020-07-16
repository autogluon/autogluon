from ..utils.misc import get_num_gpus
from .dataset import load_pandas_df, random_split_train_val, TabularDataset
from .estimators.basic_v1 import BertForTextPredictionBasic


class AutoNLP:
    @staticmethod
    def fit(train_data,
            valid_data=None,
            feature_columns=None,
            label=None,
            valid_ratio=0.15,
            exp_dir='./autonlp',
            stop_metric=None,
            eval_metrics=None,
            log_metrics=None,
            time_limits=5 * 60 * 60,
            num_gpus=None,
            hyperparameters=None):
        """

        Parameters
        ----------
        train_data
            Training dataset
        valid_data
            Validation dataset
        feature_columns
            The feature columns
        label
            Name of the label column
        valid_ratio
            Valid ratio
        exp_dir
            The experiment directory
        stop_metric
            Stop metric for model selection
        eval_metrics
            How you may potentially evaluate the model
        log_metrics
            The logging metrics
        time_limits
            The time limits.
        num_gpus
            The number of GPUs to use for the fit job. By default, we will
        hyperparameters
            The hyper-parameters of the fit function. It will include the configuration of
            the search space.

        Returns
        -------
        estimator
            An estimator object
        """
        train_data = load_pandas_df(train_data)
        if label is None:
            # Perform basic label inference
            if 'label' in train_data.columns:
                label = 'label'
            elif 'score' in train_data.columns:
                label = 'score'
            else:
                label = train_data.columns[-1]
        if feature_columns is None:
            used_columns = train_data.columns
            feature_columns = [ele for ele in used_columns if ele is not label]
        else:
            if isinstance(feature_columns, str):
                feature_columns = [feature_columns]
            used_columns = feature_columns + [label]
        train_data = TabularDataset(train_data,
                                    columns=used_columns,
                                    label_columns=label)
        column_properties = train_data.column_properties
        if valid_data is None:
            train_data, valid_data = random_split_train_val(train_data.table,
                                                            valid_ratio=valid_ratio)
            train_data = TabularDataset(train_data,
                                        columns=used_columns,
                                        column_properties=column_properties)
        else:
            valid_data = load_pandas_df(valid_data)
        valid_data = TabularDataset(valid_data,
                                    columns=used_columns,
                                    column_properties=column_properties)
        if num_gpus is None:
            num_gpus = get_num_gpus()
        if 'search_space' in hyperparameters:
            search_space = hyperparameters['search_space']
        cfg = BertForTextPredictionBasic.get_cfg()
        cfg.defrost()
        if exp_dir is not None:
            cfg.MISC.exp_dir = exp_dir
        if log_metrics is not None:
            cfg.LEARNING.log_metrics = log_metrics
        if stop_metric is not None:
            cfg.LEARNING.stop_metric = stop_metric
        cfg.freeze()
        estimator = BertForTextPredictionBasic(cfg)
        estimator.fit(train_data=train_data, valid_data=valid_data,
                      feature_columns=feature_columns,
                      label=label)
        return estimator

    @staticmethod
    def load(dir_path):
        """

        Parameters
        ----------
        dir_path

        Returns
        -------
        model
            The loaded model
        """
        BertForTextPredictionBasic.load(dir_path)
