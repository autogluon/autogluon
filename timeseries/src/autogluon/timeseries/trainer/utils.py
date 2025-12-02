import logging

logger = logging.getLogger("autogluon.timeseries.trainer")


def log_scores_and_times(
    val_score: float | None,
    fit_time: float | None,
    predict_time: float | None,
    eval_metric_name: str,
):
    if val_score is not None:
        logger.info(f"\t{val_score:<7.4f}".ljust(15) + f"= Validation score ({eval_metric_name})")
    if fit_time is not None:
        logger.info(f"\t{fit_time:<7.2f} s".ljust(15) + "= Training runtime")
    if predict_time is not None:
        logger.info(f"\t{predict_time:<7.2f} s".ljust(15) + "= Validation (prediction) runtime")
