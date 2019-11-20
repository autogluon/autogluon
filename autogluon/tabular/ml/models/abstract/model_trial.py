import logging, random, pickle
import numpy as np

from autogluon.core import * 
# from autogluon.tabular.utils.loaders.load_pd import load

logger = logging.getLogger(__name__) # TODO: Currently unused

@autogluon_register_args()
def model_trial(args, reporter):
    """ Training script for hyperparameter evaluation of an arbitrary model that subclasses AbstractModel.
        
        Notes:
            - Model object itself must be passed as kwarg: model
            - All model hyperparameters must be stored in model.params dict that may contain special keys such as:
                'seed_value' to ensure reproducibility
                'num_threads', 'num_gpus' to set specific resources in model.fit()
            - model.save() must have return_filename, file_prefix, directory options
    """
    trial_id = args.task_id # Note may not start at 0 if HPO has been run for other models with same scheduler
    directory = args.directory
    file_prefix = "trial_"+str(trial_id)+"_" # append to all file names created during this trial. Do NOT change!
    model = args.pop('model') # the model object must be passed into model_trial() here
    model.params = model.params.copy() # all hyperparameters must be stored in this dict
    model.params.update(args)
    X_train, Y_train = pickle.load(open(directory+args.dataset_train_filename, 'rb'))
    dataset_val_filename = args.get('dataset_val_filename', None)
    if dataset_val_filename is None:
        raise NotImplementedError("must provide validation data for model_trial()")
    X_val, Y_val = pickle.load(open(directory+dataset_val_filename, 'rb'))
    eval_metric = model.get_eval_metric()
    if 'seed_value' in model.params:
        random.seed(model.params['seed_value'])
        np.random.seed(model.params['seed_value'])
    
    model.fit(X_train, Y_train, X_val, Y_val)
    val_perf = model.score(X_val, Y_val)
    trial_model_file = model.save(file_prefix=file_prefix, directory=directory, return_filename=True)
    reporter(epoch=0, validation_performance=val_perf, directory=directory, file_prefix=file_prefix, 
             trial_model_file=trial_model_file) #  auxiliary_info = ?)

