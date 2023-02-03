import logging
from abc import ABC, abstractclassmethod, abstractmethod
from typing import Optional

from torch import nn

from ...constants import AUTOMM, LABEL

logger = logging.getLogger(AUTOMM)


class AbstractMultimodalFusionModel(ABC, nn.Module):
    """
    An abstract class to fuse different models' features (single-modal and multimodal).
    """

    def __init__(
        self,
        prefix: str,
        models: list,
        loss_weight: Optional[float] = None,
    ):
        super().__init__()

        self.prefix = prefix
        self.loss_weight = loss_weight
        self.model = nn.ModuleList(models)

    @property
    @abstractmethod
    def label_key(self):
        pass

    @abstractmethod
    def forward(self, *args, **kwargs):
        pass

    def get_layer_ids(
        self,
    ):
        """
        Assign an id to each layer. Layer ids will be used in layer-wise lr decay.
        Basically, id gradually increases when going from the output end to
        the input end.

        It assumes that each individual model has the "name_to_id" attribute storing
        the already computed model's layer ids. This function only collects those layer ids.
        It also add prefixes for each model's parameter names since the fusion model wraps
        those individual models, making the name scope changed. Configuring the optimizer
        requires a full name of each parameter.

        The layers defined in this class, e.g., head, adapter,
        and, fusion_mlp, have id 0.

        Returns
        -------
        A dictionary mapping the layer names (keys) to their ids (values).
        """
        model_prefix = "model"
        names = [n for n, _ in self.named_parameters()]

        outer_layer_names = [n for n in names if not n.startswith(model_prefix)]
        name_to_id = {}
        logger.debug(f"outer layers are treated as head: {outer_layer_names}")
        for n in outer_layer_names:
            name_to_id[n] = 0

        for i, per_model in enumerate(self.model):
            per_model_prefix = f"{model_prefix}.{i}"
            if not hasattr(per_model, "name_to_id"):
                raise ValueError(f"name_to_id attribute is missing in model: {per_model.__class__.__name__}")
            for n, layer_id in per_model.name_to_id.items():
                full_n = f"{per_model_prefix}.{n}"
                name_to_id[full_n] = layer_id

        # double check each parameter has been assigned an id
        for n in names:
            assert n in name_to_id

        return name_to_id
