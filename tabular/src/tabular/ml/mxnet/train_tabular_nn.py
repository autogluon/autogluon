import contextlib, shutil, tempfile, math, random, warnings, os, json, time
import logging
import os
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
from sklearn.preprocessing import StandardScaler, OneHotEncoder, QuantileTransformer, FunctionTransformer

# TODO these files should be moved eventually:
from tabular.utils.loaders import load_pkl
from tabular.ml.models.abstract_model import AbstractModel
from tabular.utils.savers import save_pkl
from tabular.ml.constants import BINARY, MULTICLASS, REGRESSION
from tabular.ml.mxnet.tabular_nn_dataset import TabularNNDataset
# import tabular.ml.mxnet.tabular_nn_model as tabNN

from autogluon.core import * 

logger = logging.getLogger(__name__) # TODO: Currently unused

@autogluon_register_args()
def train_tabularNN(args, reporter):
    """ Training function used during HPO """
    tabNN = args.tabNN
    tabNN.params.update(args) # Set params dict object == to args to explore in this trial.
    trial_id = args.task_id
    directory = args.directory
    file_prefix = "trial_"+str(trial_id)+"_" # append to all file names created during this trial. Do NOT change!
    trial_temp_file_name = directory + file_prefix + tabNN.temp_file_name # temporary file used throughout this trial
    # Load datasets:
    train_fileprefix = args.train_fileprefix
    test_fileprefix = args.test_fileprefix
    train_dataset = TabularNNDataset.load(train_fileprefix)
    test_dataset = TabularNNDataset.load(test_fileprefix)
    # Define network:
    net = tabNN.get_net(train_dataset)
    ctx = tabNN.ctx
    # Set seed:
    seed_value = args.seed_value # Set seed
    if seed_value is not None:
        random.seed(seed_value)
        np.random.seed(seed_value)
        mx.random.seed(seed_value)
    # Initialize:
    tabNN.model.collect_params().initialize(ctx=ctx)
    tabNN.model.hybridize()
    # TODO: Not used for now, we always setup the trainer:  if setup_trainer:
    tabNN.setup_trainer()
    best_val_metric = -np.inf  # higher = better
    val_metric = None
    best_val_epoch = 0
    best_train_epoch = 0  # epoch with best training loss so far
    best_train_loss = np.inf  # smaller = better
    max_epochs = args.max_epochs
    loss_scaling_factor = 1.0  # we divide loss by this quantity to stabilize gradients
    if tabNN.problem_type == REGRESSION: 
        if tabNN.metric_map[REGRESSION] == 'MAE':
            loss_scaling_factor = np.std(train_dataset.dataset._data[train_dataset.label_index])/5.0 + EPS # std-dev of labels
        elif tabNN.metric_map[REGRESSION] == 'Rsquared':
            loss_scaling_factor = np.var(train_dataset.dataset._data[train_dataset.label_index])/5.0 + EPS # variance of labels
    for e in range(max_epochs):
        cumulative_loss = 0
        for batch_idx, data_batch in enumerate(train_dataset.dataloader):
            data_batch = train_dataset.format_batch_data(data_batch, ctx)
            with autograd.record():
                output = tabNN.model(data_batch)
                labels = data_batch['label']
                loss = tabNN.loss_func(output, labels) / loss_scaling_factor
            loss.backward()
            tabNN.optimizer.step(labels.shape[0])
            cumulative_loss += nd.sum(loss).asscalar()
        train_loss = cumulative_loss/float(train_dataset.num_examples) # training loss this epoch
        if test_dataset is not None:
            val_metric = tabNN.evaluate_metric(test_dataset) # Evaluate after each epoch
        if test_dataset is None or val_metric >= best_val_metric: # assume model is getting better while validation accuracy remains the same.
            best_val_metric = val_metric
            best_val_epoch = e
            tabNN.model.save_parameters(trial_temp_file_name)
        # print("Epoch %s.  Train loss: %s, Val %s: %s" % (e, train_loss, tabNN.metric_map[tabNN.problem_type], val_metric))
        reporter(epoch=e, validation_performance=val_metric, train_loss=train_loss) # Higher val_metric = better
        if e - best_val_epoch > args.epochs_wo_improve:
            break
    tabNN.model.load_parameters(trial_temp_file_name) # Revert back to best model
    final_val_metric = tabNN.evaluate_metric(test_dataset)
    modelobj_file, netparams_file = tabNN.save(file_prefix=file_prefix, directory=directory, return_name=True)
    # print("Best model found in epoch %d. Val %s: %s" % (best_val_epoch, tabNN.metric_map[tabNN.problem_type], final_val_metric))
    # add fake epoch at the end containg performance of early-stopped model.  Higher val_metric = better
    reporter(epoch= e+1, validation_performance = final_val_metric, directory = directory, file_prefix = file_prefix,
            modelobj_file = modelobj_file, netparams_file = netparams_file)
    
    # TODO: how to keep track of which filename corresponds to which performance?

