from pathlib import Path
from typing import Union

import torch


class PipelineFactory:
    dtypes = {
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "float64": torch.float64,
    }

    @classmethod
    def get_pipeline(
        cls,
        pretrained_model_name_or_path: Union[str, Path],
        *model_args,
        force=False,
        **kwargs,
    ):
        """Load the model, either from a local path or from the HuggingFace Hub.
        Supports the same arguments as ``AutoConfig`` and ``AutoModel``
        from ``transformers``.

        When a local path is provided, supports both a folder or a .tar.gz archive.
        """
        from chronos import ChronosBoltPipeline, ChronosPipeline  # TODO: Chronos2Pipeline
        from transformers import AutoConfig

        pipeline_classes = {
            "ChronosPipeline": ChronosPipeline,
            "ChronosBoltPipeline": ChronosBoltPipeline,
        }

        kwargs.setdefault("resume_download", None)  # silence huggingface_hub warning
        if str(pretrained_model_name_or_path).startswith("s3://"):
            from .utils import cache_model_from_s3

            local_model_path = cache_model_from_s3(str(pretrained_model_name_or_path), force=force)
            return cls.get_pipeline(local_model_path, *model_args, **kwargs)

        torch_dtype = kwargs.get("torch_dtype", "auto")
        if torch_dtype != "auto" and isinstance(torch_dtype, str):
            kwargs["torch_dtype"] = cls.dtypes[torch_dtype]

        config = AutoConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
        is_valid_config = hasattr(config, "chronos_pipeline_class") or hasattr(config, "chronos_config")

        if not is_valid_config:
            raise ValueError("Not a Chronos config file")

        pipeline_class_name = getattr(config, "chronos_pipeline_class", "ChronosPipeline")
        class_ = pipeline_classes.get(pipeline_class_name, None)
        if class_ is None:
            raise ValueError(f"Trying to load unknown pipeline class: {pipeline_class_name}")

        return class_.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
