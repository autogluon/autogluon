from tabular.ml.constants import BINARY, MULTICLASS, REGRESSION


def get_param_baseline(problem_type, num_classes=None):
    if problem_type == BINARY:
        return get_param_binary_baseline()
    elif problem_type == MULTICLASS:
        return get_param_multiclass_baseline()
    elif problem_type == REGRESSION:
        return get_param_regression_baseline()
    else:
        return get_param_binary_baseline()


def get_param_multiclass_baseline():
    params = {
        'nn.tabular.ps': [0.1],
        'nn.tabular.emb_drop': 0.1,
        'nn.tabular.bs': 2048,
        'nn.tabular.lr': 1e-2,
        'nn.tabular.epochs': 30,
        'nn.tabular.metric': 'accuracy',
    }
    return params


def get_param_binary_baseline():
    params = get_param_multiclass_baseline()
    return params


def get_param_regression_baseline():
    params = {
        'nn.tabular.ps': [0.1],
        'nn.tabular.emb_drop': 0.1,
        'nn.tabular.bs': 2048,
        'nn.tabular.lr': 1e-2,
        'nn.tabular.epochs': 30,
        'nn.tabular.metric': 'mean_absolute_error',
    }
    return params


def get_nlp_param_baseline():
    params = {
        'bs': 64,  # Batch size - use maximum possible as GPU allowing
        'bptt': 8 * 9,  # Back propagation through time depth in LSTM; use multiples of 8 if fp16 training is used
        'feature_field': 'text',  # Field to use for language model training
        'max_vocab': 60000,  # Maximum vocabulary size
        'drop_mult': 0.5,  # AWD LSTM dropout multiplier (regularization) < HYPERPARAMETER
        'wd': 0.1,  # Weight decay (regularization); good values are 1e-2 and 1e-1 < HYPERPARAMETER
        'test_split_ratio_pct': 0.1,  # Fraction of train+test to use for language model training validation
        'encoders.pointer.location': './encoders.pointer',  # Location of pointer for LM models so other models can use encoders

        # Language Model training cycle parameters
        'training.lm.pretraining.epochs': 1,  # < HYPERPARAMETER, but usually kept as 1
        'training.lm.pretraining.lr': 2e-2,  # < HYPERPARAMETER
        'training.lm.finetune.epochs': 10,  # < HYPERPARAMETER
        'training.lm.finetune.lr': 2e-3,  # < HYPERPARAMETER
        'metric': 'accuracy',

        # Classifier training cycle parameters: epochs on staged layers unfreezing
        'training.classifier.l0.epochs': 1,  # on freeze(-1) < HYPERPARAMETER
        'training.classifier.l1.epochs': 1,  # on freeze(-2) < HYPERPARAMETER
        'training.classifier.l2.epochs': 1,  # on freeze(-3) < HYPERPARAMETER
        'training.classifier.l3.epochs': 2,  # on unfreeze   < HYPERPARAMETER
    }

    # Classifier training cycle parameters: learning rate schedule with staged layers unfreezing
    lr = 1e-1  # on freeze(-1)
    params['training.classifier.l0.lr'] = lr  # < HYPERPARAMETER
    lr /= 2  # on freeze(-2)
    params['training.classifier.l1.lr'] = slice(lr / (2.6 ** 4), lr)  # < HYPERPARAMETER
    lr /= 2  # on freeze(-3)
    params['training.classifier.l2.lr'] = slice(lr / (2.6 ** 4), lr)  # < HYPERPARAMETER
    lr /= 5  # on unfreeze
    params['training.classifier.l3.lr'] = slice(lr / (2.6 ** 4), lr)  # < HYPERPARAMETER

    return params
