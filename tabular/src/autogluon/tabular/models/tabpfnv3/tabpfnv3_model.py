from __future__ import annotations

from ..tabpfnv2.tabpfnv2_6_model import TabPFNv26Model


class TabPFNv3Model(TabPFNv26Model):
    """TabPFN-3 version.

    Requires `tabpfn>=8.1` for the v3 checkpoints.
    """

    ag_key = "TABPFN-3"
    ag_name = "TabPFN-3"

    default_classification_model: str | None = "tabpfn-v3-classifier-v3_default.ckpt"
    default_regression_model: str | None = "tabpfn-v3-regressor-v3_default.ckpt"
