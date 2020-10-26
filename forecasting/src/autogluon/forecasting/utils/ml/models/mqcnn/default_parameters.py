def get_default_parameters():
    """
    These default values are used for quick test
    """
    hps = {
        "seed": 42,
        "quantiles": [0.5, 0.9],
        "epochs": 3,
        "num_batches_per_epoch": 32,
    }
    return hps
