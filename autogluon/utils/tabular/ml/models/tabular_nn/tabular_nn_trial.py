import random, logging
import numpy as np
import mxnet as mx
from mxnet import nd, autograd

from ......core import args
from .tabular_nn_dataset import TabularNNDataset

logger = logging.getLogger(__name__) # TODO: Currently unused
EPS = 10e-8 # small number

@args()
def tabular_nn_trial(args, reporter):
    """ Training and evaluation function used during a single trial of HPO """
    tabNN = args.pop('tabNN')
    tabNN.params = tabNN.params.copy() # duplicate to make sure there are no remaining pointers across trials.
    tabNN.params.update(args) # Set params dict object == to args to explore in this trial.
    trial_id = args.task_id # Note may not start at 0 if HPO has been run for other models with same scheduler
    directory = args.directory
    file_prefix = "trial_"+str(trial_id)+"_" # append to all file names created during this trial. Do NOT change!
    trial_temp_file_name = directory + file_prefix + tabNN.temp_file_name # temporary file used throughout this trial
    # Load datasets:
    train_fileprefix = args.train_fileprefix
    test_fileprefix = args.test_fileprefix # cannot be None
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
    best_val_metric = -np.inf  # we always assume higher = better
    val_metric = None
    best_val_epoch = 0
    best_train_epoch = 0  # epoch with best training loss so far
    best_train_loss = np.inf  # smaller = better
    num_epochs = args.num_epochs
    if test_dataset is not None:
        y_test = test_dataset.get_labels()
    
    loss_scaling_factor = 1.0 # we divide loss by this quantity to stabilize gradients
    loss_torescale = [key for key in tabNN.rescale_losses if isinstance(tabNN.loss_func, key)]
    if len(loss_torescale) > 0:
        loss_torescale = loss_torescale[0]
        if tabNN.rescale_losses[loss_torescale] == 'std':
            loss_scaling_factor = np.std(train_dataset.get_labels())/5.0 + EPS # std-dev of labels
        elif tabNN.rescale_losses[loss_torescale] == 'var':
            loss_scaling_factor = np.var(train_dataset.get_labels())/5.0 + EPS # variance of labels
        else:
            raise ValueError("Unknown loss-rescaling type %s specified for loss_func==%s" % (tabNN.rescale_losses[loss_torescale],tabNN.loss_func))
    # Training Loop:
    for e in range(num_epochs):
        cumulative_loss = 0
        for batch_idx, data_batch in enumerate(train_dataset.dataloader):
            data_batch = train_dataset.format_batch_data(data_batch, ctx)
            with autograd.record():
                output = tabNN.model(data_batch)
                labels = data_batch['label']
                loss = tabNN.loss_func(output, labels) / loss_scaling_factor
            loss.backward()
            tabNN.optimizer.step(labels.shape[0])
            cumulative_loss += loss.sum()
        train_loss = cumulative_loss/float(train_dataset.num_examples) # training loss this epoch
        if test_dataset is not None:
            # val_metric = tabNN.evaluate_metric(test_dataset) # Evaluate after each epoch
            val_metric = tabNN.score(X=test_dataset, y=y_test)
        if test_dataset is None or val_metric >= best_val_metric: # assume model is getting better while validation accuracy remains the same.
            best_val_metric = val_metric
            best_val_epoch = e
            tabNN.model.save_parameters(trial_temp_file_name)
        # print("Epoch %s.  Train loss: %s, Val %s: %s" % (e, train_loss, tabNN.eval_metric_name, val_metric))
        reporter(epoch=e, validation_performance=val_metric, train_loss=float(train_loss.asscalar())) # Higher val_metric = better
        if e - best_val_epoch > args.epochs_wo_improve:
            break
    tabNN.model.load_parameters(trial_temp_file_name) # Revert back to best model
    # final_val_metric = tabNN.evaluate_metric(test_dataset)
    final_performance = tabNN.score(X=test_dataset, y=y_test)
    modelobj_file, netparams_file = tabNN.save(file_prefix=file_prefix, directory=directory, return_name=True)
    # print("Best model found in epoch %d. Val %s: %s" % (best_val_epoch, tabNN.eval_metric_name, final_val_metric))
    # add fake epoch at the end containg performance of early-stopped model.  Higher val_metric = better
    reporter(epoch= e+1, validation_performance = final_performance, directory = directory, file_prefix = file_prefix,
             modelobj_file = modelobj_file, netparams_file = netparams_file) # unused in reporter, only directory+file_prefix required to load model.
    
    # TODO: how to keep track of which filename corresponds to which performance?
