
from autogluon.core.utils.early_stopping import AdaptiveES, ES_CLASS_MAP


# TODO: Add more strategies
def get_early_stopping_rounds(num_rows_train, strategy='auto', min_patience=10, max_patience=150, min_rows=10000):
    if isinstance(strategy, (tuple, list)):
        strategy = list(strategy)
        if isinstance(strategy[0], str):
            if strategy[0] in ES_CLASS_MAP:
                strategy[0] = ES_CLASS_MAP[strategy[0]]
            else:
                raise AssertionError(f'unknown early stopping strategy: {strategy}')
        return strategy

    """Gets early stopping rounds"""
    if strategy == 'auto':
        strategy = 'simple'

    modifier = 1 if num_rows_train <= min_rows else min_rows / num_rows_train
    simple_early_stopping_rounds = max(
        round(modifier * max_patience),
        min_patience,
    )
    if strategy == 'simple':
        return simple_early_stopping_rounds
    elif strategy == 'adaptive':
        return AdaptiveES, dict(adaptive_offset=min_patience, min_patience=simple_early_stopping_rounds)
    else:
        raise AssertionError(f'unknown early stopping strategy: {strategy}')
