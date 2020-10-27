from autogluon.core import Real

# TODO: I don't see a need to split search space by problem type.
#  Is there a reason to at this point?
def get_default_searchspace():
    params = {
        'lr': Real(5e-5, 5e-3, log=True)
    }

    return params.copy()
