from autogluon.core.utils.early_stopping import ES_CLASS_MAP, AdaptiveES


# TODO: Add more strategies
def get_early_stopping_rounds(
    num_rows_train: int, strategy="auto", min_offset: int = 20, max_offset: int = 300, min_rows: int = 10000
):
    """

    Parameters
    ----------
    num_rows_train : int
        The number of rows in the training data.
        # FIXME: Better to be number of rows in validation data? Should test with simulator
    strategy : str or tuple, default "auto"
        If str, one of ["auto", "simple", "adaptive"]
        If tuple, contains two elements. The first element is the early stopping class, and the second is the init kwargs.
        # FIXME: If tuple, the min_offset and max_offset logic is skipped!
    min_offset : int, default 20
        The minimum offset (b) value for patience.
    max_offset : int, default 300
        The maximum offset (b) value for patience.
    min_rows : int, default 10000
        The training row count floor. The selected offset will be `max_offset` if `num_rows_train <= min_rows`.
        Else, the selected offset will be `max(min_offset, round(max_offset * min_rows / num_rows_train))`.

    """
    if isinstance(strategy, (tuple, list)):
        strategy = list(strategy)
        if isinstance(strategy[0], str):
            if strategy[0] in ES_CLASS_MAP:
                strategy[0] = ES_CLASS_MAP[strategy[0]]
            else:
                raise AssertionError(f"unknown early stopping strategy: {strategy}")
        return strategy

    """Gets early stopping rounds"""
    if strategy == "auto":
        strategy = "simple"

    modifier = 1 if num_rows_train <= min_rows else min_rows / num_rows_train
    simple_early_stopping_rounds = max(
        round(modifier * max_offset),
        min_offset,
    )
    if strategy == "simple":
        return simple_early_stopping_rounds
    elif strategy == "adaptive":
        return AdaptiveES, dict(adaptive_offset=min_offset, min_patience=simple_early_stopping_rounds)
    else:
        raise AssertionError(f"unknown early stopping strategy: {strategy}")
