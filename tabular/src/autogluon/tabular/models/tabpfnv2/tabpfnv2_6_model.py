from .tabpfnv2_5_model import RealTabPFNv25Model


class TabPFNv26Model(RealTabPFNv25Model):
    """TabPFN-2.6 version."""

    ag_key = "TABPFN-2.6"
    ag_name = "TabPFN-2.6"

    fixed_random_state: int = 0
    """We found that the validation score is misleading for TabPFN, when one uses a
    different random state for the refit model than for models fit during CV.
    This is because TabPFN's random state determines the preprocessing of TabPFN
    """

    default_classification_model: str | None = "tabpfn-v2.6-classifier-v2.6_default.ckpt"
    default_regression_model: str | None = "tabpfn-v2.6-regressor-v2.6_default.ckpt"

    def _get_default_auxiliary_params(self) -> dict:
        default_auxiliary_params = super()._get_default_auxiliary_params()
        default_auxiliary_params.update(
            {
                "max_rows": 100_000,
                "max_features": 2500,
                "max_classes": 10,
                "model_telemetry": False,
            }
        )
        return default_auxiliary_params

    @staticmethod
    def extra_checkpoints_for_tuning(problem_type: str) -> list[str]:
        """The list of checkpoints to use for hyperparameter tuning."""
        raise NotImplementedError(
            "We did not benchmark more checkpoints or tuning."
        )
