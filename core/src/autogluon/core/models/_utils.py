
# TODO: Add more strategies
#  - Adaptive early stopping: adjust rounds during model training
def get_early_stopping_rounds(num_rows_train, strategy='auto', min_rounds=10, max_rounds=150, min_rows=10000):
    """Gets early stopping rounds"""
    if strategy == 'auto':
        modifier = 1 if num_rows_train <= min_rows else min_rows / num_rows_train
        early_stopping_rounds = max(
            round(modifier * max_rounds),
            min_rounds,
        )
    else:
        raise AssertionError(f'unknown early stopping strategy: {strategy}')
    return early_stopping_rounds
