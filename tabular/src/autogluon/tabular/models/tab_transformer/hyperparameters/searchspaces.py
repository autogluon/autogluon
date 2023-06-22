from autogluon.common import space


# TODO: May have to split search space's by problem type. Not necessary right now.
def get_default_searchspace():
    params = {
        "lr": space.Real(5e-5, 5e-3, default=1e-3, log=True),
        "weight_decay": space.Real(1e-6, 5e-2, default=1e-6, log=True),
        "p_dropout": space.Categorical(0.1, 0, 0.5),
        "n_heads": space.Categorical(8, 4),
        "hidden_dim": space.Categorical(128, 32, 64, 256),
        "n_layers": space.Categorical(2, 1, 3, 4, 5),
        "feature_dim": space.Int(8, 128, default=64),
        "num_output_layers": space.Categorical(1, 2),
    }

    return params.copy()
