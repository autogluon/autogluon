import os
import warnings
from collections import defaultdict, namedtuple
from typing import Dict, List, Optional, Union

import pandas as pd
import torch

from ..constants import HF_TEXT, MMDET_IMAGE, TEXT, TIMM_IMAGE
from ..models.fusion import AbstractMultimodalFusionModel
from ..models.huggingface_text import HFAutoModelForTextPrediction
from ..models.mmdet_image import MMDetAutoModelForObjectDetection
from ..models.timm_image import TimmAutoModelForImagePrediction
from .environment import compute_num_gpus, get_precision_context, infer_precision, move_to_device
from .inference import process_batch
from .onnx import get_onnx_input


class ExportMixin:
    def dump_model(self, save_path: Optional[str] = None):
        """
        Save model weights and config to local directory.
        Model weights are saved in file `pytorch_model.bin` (timm, hf) or '<ckpt_name>.pth' (mmdet);
        Configs are saved in file `config.json` (timm, hf) or  '<ckpt_name>.py' (mmdet).

        Parameters
        ----------
        path : str
            Path to directory where models and configs should be saved.
        """

        if not save_path:
            save_path = self._save_path if self._save_path else "./"

        supported_models = {
            TIMM_IMAGE: TimmAutoModelForImagePrediction,
            HF_TEXT: HFAutoModelForTextPrediction,
            MMDET_IMAGE: MMDetAutoModelForObjectDetection,
        }

        models = defaultdict(list)
        # TODO: simplify the code
        if isinstance(self._model, AbstractMultimodalFusionModel) and isinstance(
            self._model.model, torch.nn.modules.container.ModuleList
        ):
            for per_model in self._model.model:
                for model_key, model_type in supported_models.items():
                    if isinstance(per_model, model_type):
                        models[model_key].append(per_model)
        else:
            for model_key, model_type in supported_models.items():
                if isinstance(self._model, model_type):
                    models[model_key].append(self._model)

        if not models:
            raise NotImplementedError(
                f"No models available for dump. Current supported models are: {supported_models.keys()}"
            )

        # get tokenizers for hf_text
        text_processors = self._data_processors.get(TEXT, {})
        tokenizers = {}
        for per_processor in text_processors:
            tokenizers[per_processor.prefix] = per_processor.tokenizer

        for model_key in models:
            for per_model in models[model_key]:
                subdir = os.path.join(save_path, per_model.prefix)
                os.makedirs(subdir, exist_ok=True)
                per_model.save(save_path=subdir, tokenizers=tokenizers)

        return save_path

    def export_onnx(
        self,
        data: Optional[pd.DataFrame] = None,
        onnx_path: Optional[str] = None,
        batch_size: Optional[int] = None,
        verbose: Optional[bool] = False,
        opset_version: Optional[int] = 16,
    ):
        """
        Export this predictor's model to ONNX file.

        Parameters
        ----------
        onnx_path
            The export path of onnx model.
        data
            Raw data used to trace and export the model.
            If this is None, will check if a processed batch is provided.
        batch_size
            The batch_size of export model's input.
            Normally the batch_size is a dynamic axis, so we could use a small value for faster export.
        verbose
            verbose flag in torch.onnx.export.
        opset_version
            opset_version flag in torch.onnx.export.
        """
        # TODO: Support CLIP
        # TODO: Add test
        import torch.jit

        from ..models.huggingface_text import HFAutoModelForTextPrediction
        from ..models.timm_image import TimmAutoModelForImagePrediction

        supported_models = (TimmAutoModelForImagePrediction, HFAutoModelForTextPrediction)
        if not isinstance(self._model, supported_models):
            raise NotImplementedError(f"export_onnx doesn't support model type {type(self._model)}")
        warnings.warn("Currently, the functionality of exporting to ONNX is experimental.")

        valid_input, dynamic_axes, default_onnx_path, batch = get_onnx_input(
            pipeline=self._problem_type, config=self._config
        )

        if not batch_size:
            batch_size = 2  # batch_size should be a dynamic_axis, so we could use a small value for faster export
        if data is not None:
            batch = self.get_processed_batch_for_deployment(
                data=data,
                valid_input=valid_input,
                onnx_tracing=True,
                batch_size=batch_size,
            )

        if not onnx_path:
            onnx_path = os.path.join(self.path, default_onnx_path)

        device_type = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(device_type)

        self._model.eval()

        strategy = "dp"  # default used in inference.
        num_gpus = compute_num_gpus(config_num_gpus=self._config.env.num_gpus, strategy=strategy)
        precision = infer_precision(
            num_gpus=num_gpus,
            precision=self._config.env.precision,
            cpu_only_warning=False,
        )
        precision_context = get_precision_context(precision=precision, device_type=device_type)

        InputBatch = namedtuple("InputBatch", self._model.input_keys)

        # Perform tracing on cpu, since we're facing an error when tracing with cuda device:
        #     ERROR: Tensor-valued Constant nodes differed in value across invocations.
        #     This often indicates that the tracer has encountered untraceable code.
        #     Comparison exception:   The values for attribute 'shape' do not match: torch.Size([]) != torch.Size([384]).
        #     from https://github.com/rwightman/pytorch-image-models/blob/3aa31f537d5fbf6be8f1aaf5a36f6bbb4a55a726/timm/models/swin_transformer.py#L112
        device = "cpu"
        num_gpus = 0
        dtype = infer_precision(
            num_gpus=num_gpus, precision=self._config.env.precision, cpu_only_warning=False, as_torch=True
        )
        for key in self._model.input_keys:
            inp = batch[key]
            # support mixed precision on floating point inputs, and leave integer inputs (for language models) untouched.
            if inp.dtype.is_floating_point:
                batch[key] = inp.to(device, dtype=dtype)
            else:
                batch[key] = inp.to(device)
        self._model.to(device)
        input_vec = InputBatch(**batch)

        with precision_context, torch.no_grad():
            traced_model = torch.jit.trace(self._model, input_vec)
        torch.jit.save(traced_model, "traced_model.pt")
        traced_model = torch.jit.load("traced_model.pt")

        torch.onnx.export(
            traced_model,
            args=input_vec,
            f=onnx_path,
            opset_version=opset_version,
            verbose=verbose,
            input_names=valid_input,
            dynamic_axes=dynamic_axes,
        )

    def get_processed_batch_for_deployment(
        self,
        data: Union[pd.DataFrame, dict],
        valid_input: Optional[List] = None,
        onnx_tracing: bool = False,
        batch_size: int = None,
        to_numpy: bool = True,
        requires_label: bool = False,
    ):
        """
        Get the processed batch of raw data given.

        Parameters
        ----------
        data
            The raw data to process
        valid_input
            Used to filter valid data. No filter happens if it is empty.
        onnx_tracing
            If the output is used for onnx tracing.
        batch_size
            The batch_size of output batch.
            If onnx_tracing, it will only output one mini-batch, and all int tensor values will be converted to long.
        to_numpy
            Output numpy array if True. Only valid if not onnx_tracing.

        Returns
        -------
        Tensor or numpy array.
        The output processed batch could be used for export/evaluate deployed model.
        """
        data, df_preprocessor, data_processors = self._on_predict_start(
            data=data,
            requires_label=requires_label,
        )

        batch = process_batch(
            data=data,
            df_preprocessor=df_preprocessor,
            data_processors=data_processors,
        )

        ret = {}
        for k in batch:
            if valid_input and k not in valid_input:
                continue
            if onnx_tracing:
                ret[k] = batch[k].long() if isinstance(batch[k], torch.IntTensor) else batch[k]
            elif to_numpy:
                ret[k] = batch[k].cpu().detach().numpy().astype(int)
            else:
                ret[k] = batch[k]
        if not onnx_tracing:
            if batch_size:
                raise NotImplementedError("We should split the batch here.")  # TODO
        return ret
