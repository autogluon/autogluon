from packaging import version

from autogluon.tabular import __version__


def test_tabular_predictor_dry_run_default_false_v140():
    ag_version = __version__
    # Delete this test after updating the method for 1.4.0.
    if version.parse(ag_version) >= version.parse("1.4.0"):
        # FutureWarning added in v1.3.0
        # Reasoning: most users probably don't care about dry_run when calling this method,
        # so it could lead to unintentional bugs / confusing downstream for
        # users if they aren't aware that they need to set dry_run=False for it to do anything.
        raise AssertionError(
            f"Verify that `TabularPredictor.delete_models` dry_run=False "
            f"is the default starting in v1.4.0. Remove None as an option."
        )
