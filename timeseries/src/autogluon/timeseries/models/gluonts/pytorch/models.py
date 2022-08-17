"""
Module including wrappers for PyTorch implementations of models in GluonTS
"""
import logging
from typing import Optional, Type

import pytorch_lightning as pl
from gluonts.core.component import from_hyperparameters
from gluonts.torch.model.estimator import PyTorchLightningEstimator as GluonTSPyTorchLightningEstimator
from gluonts.torch.model.deepar import DeepAREstimator

# TODO: enable in GluonTS v0.10
# from gluonts.torch.model.mqf2 import MQF2MultiHorizonEstimator
# from gluonts.torch.model.simple_feedforward import SimpleFeedForwardEstimator

from autogluon.common.utils.log_utils import set_logger_verbosity
from autogluon.core.utils import warning_filter
from autogluon.timeseries import TimeSeriesDataFrame
from autogluon.timeseries.utils.warning_filters import disable_root_logger

from ..abstract_gluonts import SimpleGluonTSDataset, AbstractGluonTSModel
from .callback import PLTimeLimitCallback


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


class DeepARPyTorchModel(AbstractGluonTSPytorchModel):
    gluonts_estimator_class: Type[GluonTSPyTorchLightningEstimator] = DeepAREstimator
