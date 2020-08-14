def get_default_parameters():
    hps = {
        "seed": 42,
        "freq": "D",
        "context_length": 5,
        "prediction_length": 3,
        "quantiles": [0.5, 0.1],
        "epochs": 3,
        "num_batches_per_epoch": 3,
    }
    return hps

