"""
Module including wrappers for PyTorch implementations of models in GluonTS
"""
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Type

from gluonts.core.component import from_hyperparameters
from gluonts.torch.model.deepar import DeepAREstimator
from gluonts.torch.model.estimator import PyTorchLightningEstimator as GluonTSPyTorchLightningEstimator
from gluonts.torch.model.predictor import PyTorchPredictor as GluonTSPyTorchPredictor

from autogluon.common.utils.log_utils import set_logger_verbosity
from autogluon.core.utils import warning_filter
from autogluon.timeseries import TimeSeriesDataFrame
from autogluon.timeseries.utils.warning_filters import disable_root_logger

from ..abstract_gluonts import AbstractGluonTSModel, SimpleGluonTSDataset
from .callback import PLTimeLimitCallback

# TODO: enable in GluonTS v0.10
# from gluonts.torch.model.mqf2 import MQF2MultiHorizonEstimator
# from gluonts.torch.model.simple_feedforward import SimpleFeedForwardEstimator


logger = logging.getLogger(__name__)
pl_loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict if "pytorch_lightning" in name]


class AbstractGluonTSPytorchModel(AbstractGluonTSModel):
    gluonts_estimator_class: Type[GluonTSPyTorchLightningEstimator]

    def _get_estimator(self) -> GluonTSPyTorchLightningEstimator:
        """Return the GluonTS Estimator object for the model"""

        # As GluonTSPyTorchLightningEstimator objects do not implement `from_hyperparameters` convenience
        # constructors, we re-implement the logic here.
        # we translate the "epochs" parameter to "max_epochs" for consistency in the AbstractGluonTSModel
        # interface

        init_args = self._get_estimator_init_args()

        trainer_kwargs = {}
        epochs = init_args.get("max_epochs", init_args.get("epochs"))
        callbacks = init_args.get("callbacks", [])

        if epochs:
            trainer_kwargs.update({"max_epochs": epochs, "callbacks": callbacks, "progress_bar_refresh_rate": 0})

        return from_hyperparameters(
            self.gluonts_estimator_class,
            trainer_kwargs=trainer_kwargs,
            **init_args,
        )

    def _fit(
        self,
        train_data: TimeSeriesDataFrame,
        val_data: Optional[TimeSeriesDataFrame] = None,
        time_limit: int = None,
        **kwargs,
    ) -> None:
        verbosity = kwargs.get("verbosity", 2)
        set_logger_verbosity(verbosity, logger=logger)
        for pl_logger in pl_loggers:
            pl_logger.setLevel(logging.ERROR if verbosity <= 3 else logging.INFO)

        if verbosity > 3:
            logger.warning(
                "GluonTS logging is turned on during training. Note that losses reported by GluonTS "
                "may not correspond to those specified via `eval_metric`."
            )

        self._check_fit_params()
        # TODO: reintroduce early stopping callbacks

        # update auxiliary parameters
        self._deferred_init_params_aux(dataset=train_data, callbacks=[PLTimeLimitCallback(time_limit)], **kwargs)

        estimator = self._get_estimator()
        with warning_filter(), disable_root_logger():
            self.gts_predictor = estimator.train(
                self._to_gluonts_dataset(train_data),
                validation_data=self._to_gluonts_dataset(val_data),
            )

    @classmethod
    def load(cls, path: str, reset_paths: bool = True, verbose: bool = True) -> "AbstractGluonTSModel":
        model = super().load(path, reset_paths, verbose)
        model.gts_predictor = GluonTSPyTorchPredictor.deserialize(Path(path) / cls.gluonts_model_path)
        return model


class DeepARPyTorchModel(AbstractGluonTSPytorchModel):
    gluonts_estimator_class: Type[GluonTSPyTorchLightningEstimator] = DeepAREstimator

    def _get_estimator_init_args(self) -> Dict[str, Any]:
        """Get GluonTS specific constructor arguments for estimator objects, an alias to
        `self._get_model_params` for better readability."""
        init_kwargs = self._get_model_params()

        # GluonTS does not handle context_length=1 well, and sets the context to only prediction_length
        # we set it to a minimum of 10 here.
        init_kwargs["context_length"] = max(10, init_kwargs.get("context_length", self.prediction_length))

        return init_kwargs
