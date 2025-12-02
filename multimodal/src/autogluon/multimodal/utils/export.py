import io
import logging
import os
import warnings
from collections import defaultdict
from typing import Dict, List, Optional, Union

import pandas as pd
import torch

from ..constants import CATEGORICAL, HF_TEXT, IMAGE_PATH, MMDET_IMAGE, NULL, NUMERICAL, TEXT, TIMM_IMAGE
from ..models.fusion import AbstractMultimodalFusionModel
from ..models.hf_text import HFAutoModelForTextPrediction
from ..models.mmdet_image import MMDetAutoModelForObjectDetection
from ..models.timm_image import TimmAutoModelForImagePrediction
from .onnx import OnnxModule, onnx_get_dynamic_axes
from .precision import infer_precision

logger = logging.getLogger(__name__)


class ExportMixin:
    def dump_model(self, save_path: Optional[str] = None):
        """
        Save model weights and config to local directory.
        Model weights are saved in file `pytorch_model.bin` (timm, hf) or '<ckpt_name>.pth' (mmdet);
        Configs are saved in file `config.json` (timm, hf) or  '<ckpt_name>.py' (mmdet).

        Parameters
        ----------
        save_path : str
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

        for model_key in models:
            for per_model in models[model_key]:
                subdir = os.path.join(save_path, per_model.prefix)
                os.makedirs(subdir, exist_ok=True)
                per_model.save(save_path=subdir)

        return save_path

    def export_onnx(
        self,
        data: Union[dict, pd.DataFrame],
        path: Optional[str] = None,
        batch_size: Optional[int] = None,
        verbose: Optional[bool] = False,
        opset_version: Optional[int] = 16,
        truncate_long_and_double: Optional[bool] = False,
    ):
        """
        Export this predictor's model to ONNX file.

        When `path` argument is not provided, the method would not save the model into disk.
        Instead, it would export the onnx model into BytesIO and return its binary as bytes.

        Parameters
        ----------
        data
            Raw data used to trace and export the model.
            If this is None, will check if a processed batch is provided.
        path : str, default=None
            The export path of onnx model. If path is not provided, the method would export model to memory.
        batch_size
            The batch_size of export model's input.
            Normally the batch_size is a dynamic axis, so we could use a small value for faster export.
        verbose
            verbose flag in torch.onnx.export.
        opset_version
            opset_version flag in torch.onnx.export.
        truncate_long_and_double: bool, default False
            Truncate weights provided in int64 or double (float64) to int32 and float32

        Returns
        -------
        onnx_path : str or bytes
            A string that indicates location of the exported onnx model, if `path` argument is provided.
            Otherwise, would return the onnx model as bytes.
        """

        import torch.jit

        from ..models.fusion.fusion_mlp import MultimodalFusionMLP
        from ..models.hf_text import HFAutoModelForTextPrediction
        from ..models.timm_image import TimmAutoModelForImagePrediction

        supported_models = (TimmAutoModelForImagePrediction, HFAutoModelForTextPrediction, MultimodalFusionMLP)
        if not isinstance(self._model, supported_models):
            raise NotImplementedError(f"export_onnx doesn't support model type {type(self._model)}")
        warnings.warn("Currently, the functionality of exporting to ONNX is experimental.")

        # Data preprocessing, loading, and filtering
        batch = self.get_processed_batch_for_deployment(
            data=data,
            onnx_tracing=True,
            batch_size=batch_size,
            truncate_long_and_double=truncate_long_and_double,
        )
        input_keys = self._model.input_keys
        input_vec = [batch[k] for k in input_keys]

        # Write to BytesIO if path argument is not provided
        if path is None:
            onnx_path = io.BytesIO()
        else:
            onnx_path = os.path.join(path, "model.onnx")
            dirname = os.path.dirname(os.path.abspath(onnx_path))
            if not os.path.exists(dirname):
                os.makedirs(dirname)

        # Infer dynamic dimensions
        dynamic_axes = onnx_get_dynamic_axes(input_keys)

        torch.onnx.export(
            self._model.eval(),
            args=tuple(input_vec),
            f=onnx_path,
            opset_version=opset_version,
            verbose=verbose,
            input_names=input_keys,
            dynamic_axes=dynamic_axes,
            dynamo=False,
        )

        if isinstance(onnx_path, io.BytesIO):
            onnx_path = onnx_path.getvalue()

        return onnx_path

    def optimize_for_inference(
        self,
        providers: Optional[Union[dict, List[str]]] = None,
    ):
        """
        Optimize the predictor's model for inference.

        Under the hood, the implementation would convert the PyTorch module into an ONNX module, so that
        we can leverage efficient execution providers in onnxruntime for faster inference.

        Parameters
        ----------
        data
            Raw data used to trace and export the model.
            If this is None, will check if a processed batch is provided.
        providers : dict or str, default=None
            A list of execution providers for model prediction in onnxruntime.

            By default, the providers argument is None. The method would generate an ONNX module that
            would perform model inference with TensorrtExecutionProvider in onnxruntime, if tensorrt
            package is properly installed. Otherwise, the onnxruntime would fallback to use CUDA or CPU
            execution providers instead.

        Returns
        -------
        onnx_module : OnnxModule
            The onnx-based module that can be used to replace predictor._model for model inference.
        """
        data_dict = {}
        for col_name, col_type in self._column_types.items():
            if col_type in [NUMERICAL, CATEGORICAL, NULL]:
                data_dict[col_name] = [0, 1]
            elif col_type == TEXT:
                data_dict[col_name] = ["some text", "some other text"]
            elif col_type in [IMAGE_PATH]:
                data_dict[col_name] = ["/not-exist-dir/xxx.jpg", "/not-exist-dir/yyy.jpg"]
            else:
                raise ValueError(f"unsupported column type: {col_type}")
        data = pd.DataFrame.from_dict(data_dict)

        onnx_module = None
        onnx_path = self.export_onnx(data=data, truncate_long_and_double=True)

        onnx_module = OnnxModule(onnx_path, providers)
        onnx_module.input_keys = self._model.input_keys
        onnx_module.prefix = self._model.prefix
        onnx_module.get_output_dict = self._model.get_output_dict

        # To use the TensorRT module for prediction, simply replace the _model in the predictor
        self._model = onnx_module

        # Evaluate and cache TensorRT engine files
        logger.info("Compiling ... (this may take a few minutes)")
        _ = self.predict(data)
        logger.info("Finished compilation!")

        return onnx_module

    def get_processed_batch_for_deployment(
        self,
        data: Union[pd.DataFrame, dict],
        onnx_tracing: bool = False,
        batch_size: int = None,
        to_numpy: bool = True,
        requires_label: bool = False,
        truncate_long_and_double: bool = False,
    ):
        """
        Get the processed batch of raw data given.

        Parameters
        ----------
        data
            The raw data to process
        onnx_tracing
            If the output is used for onnx tracing.
        batch_size
            The batch_size of output batch.
            If onnx_tracing, it will only output one mini-batch, and all int tensor values will be converted to long.
        to_numpy
            Output numpy array if True. Only valid if not onnx_tracing.
        require_label
            Whether do we put label data into the output batch

        Returns
        -------
        Tensor or numpy array.
        The output processed batch could be used for export/evaluate deployed model.
        """
        data = self.data_to_df(data=data)
        column_types = self.infer_column_types(
            column_types=self._column_types,
            data=data,
            is_train=False,
        )
        df_preprocessor = self.get_df_preprocessor_per_run(
            df_preprocessor=self._df_preprocessor,
            data=data,
            column_types=column_types,
            is_train=False,
        )
        if self._fit_called:
            df_preprocessor._column_types = self.update_image_column_types(data=data)
        data_processors = self.get_data_processors_per_run(
            data_processors=self._data_processors,
            requires_label=requires_label,
            is_train=False,
        )

        batch = self.process_batch(
            data=data,
            df_preprocessor=df_preprocessor,
            data_processors=data_processors,
        )

        input_keys = self._model.input_keys

        # Perform tracing on cpu
        device_type = "cpu"
        num_gpus = 0
        strategy = "dp"  # default used in inference.
        device = torch.device(device_type)
        dtype = infer_precision(
            num_gpus=num_gpus, precision=self._config.env.precision, cpu_only_warning=False, as_torch=True
        )

        # Move model data to the specified device
        for key in input_keys:
            inp = batch[key]
            # support mixed precision on floating point inputs, and leave integer inputs (for language models) untouched.
            if inp.dtype.is_floating_point:
                batch[key] = inp.to(device, dtype=dtype)
            else:
                batch[key] = inp.to(device)
        self._model.to(device)

        # Truncate input data types for TensorRT (only support: bool, int32, half, float)
        if truncate_long_and_double:
            for k in batch:
                if batch[k].dtype == torch.int64:
                    batch[k] = batch[k].to(torch.int32)

        # Data filtering
        ret = {}
        for k in batch:
            if input_keys and k not in input_keys:
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
