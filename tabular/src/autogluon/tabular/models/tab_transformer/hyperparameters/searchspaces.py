from autogluon.core import Real

# TODO: May have to split search space's by problem type. Not necessary now.
def get_default_searchspace():
    params = {
        'lr': Real(5e-5, 5e-3, log=True)
    }

    return params.copy()
