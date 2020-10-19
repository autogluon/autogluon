def get_default_parameters():
        hps = {
            "seed": 42,
            "freq": "D",
            "context_length": 5,
            "prediction_length": 19,
            "quantiles": [0.9],
            "epochs": 3,
            "num_batches_per_epoch": 32,
        }
        return hps
