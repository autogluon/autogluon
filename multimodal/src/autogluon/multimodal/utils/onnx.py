import logging
import os
from typing import Dict, List, Optional, Tuple, Union

from torch import tensor

from ..constants import AUTOMM, FEATURE_EXTRACTION, MULTICLASS

logger = logging.getLogger(__name__)

# TODO: Try a better workaround to lazy import tensorrt package.
tensorrt_imported = False
if not tensorrt_imported:
    try:
        import tensorrt  # Unused but required by TensorrtExecutionProvider

        tensorrt_imported = True
    except:
        # We silently omit the import failure here to avoid overwhelming warning messages in case of multi-gpu.
        tensorrt_imported = False


def onnx_get_dynamic_axes(input_keys: List[str]):
    dynamic_axes = {}
    for k in input_keys:
        if "token_ids" in k or "segment_ids" in k:
            dynamic_axes[k] = {0: "batch_size", 1: "seq_length"}
        elif "valid_length" in k or k.startswith("numerical") or k.startswith("timm_image"):
            dynamic_axes[k] = {0: "batch_size"}

    return dynamic_axes


def get_provider_name(provider_config: Union[str, tuple]) -> str:
    if isinstance(provider_config, tuple):
        provider_name = provider_config[0]
    else:
        assert isinstance(provider_config, str), "input provider config is expected to be either str or tuple"
        provider_name = provider_config
    return provider_name


class OnnxModule(object):
    """
    OnnxModule is as a replacement of torch.nn.Module for running forward pass with onnxruntime.

    The module can be generated with MultiModalPredictor.export_tensorrt(),
    so that we can predict with TensorRT by simply replacing predictor._model with OnnxModule.
    """

    def __init__(self, onnx_path: Union[str, bytes], providers: Optional[Union[dict, List[str]]] = None):
        """
        Parameters
        ----------
        onnx_path : str or bytes
            The file path (or bytes) of the onnx model that need to be executed in onnxruntime.
        providers : dict or str, default=None
            A list of execution providers for model prediction in onnxruntime.
        """
        import onnx
        import onnxruntime as ort

        if isinstance(onnx_path, bytes):
            onnx_model = onnx.load_model_from_string(onnx_path)
        else:
            if not os.path.exists(onnx_path):
                raise FileNotFoundError(f"failed to located onnx file at {onnx_path}")

            logger.info("Loading ONNX file from path {}...".format(onnx_path))
            onnx_model = onnx.load(onnx_path)

        if providers is None:
            if isinstance(onnx_path, str):
                dirname = os.path.dirname(os.path.abspath(onnx_path))
                cache_path = os.path.join(dirname, "model_trt")
            else:
                cache_path = None
            providers = [
                (
                    "TensorrtExecutionProvider",
                    {
                        "device_id": 0,
                        "trt_max_workspace_size": 2147483648,
                        "trt_fp16_enable": True,
                        "trt_engine_cache_path": cache_path,
                        "trt_engine_cache_enable": True,
                    },
                ),
                (
                    "CUDAExecutionProvider",
                    {
                        "device_id": 0,
                        "arena_extend_strategy": "kNextPowerOfTwo",
                        "gpu_mem_limit": 2 * 1024 * 1024 * 1024,
                        "cudnn_conv_algo_search": "EXHAUSTIVE",
                        "do_copy_in_default_stream": True,
                    },
                ),
                ("CPUExecutionProvider", {}),
            ]

        if len(providers) == 1 and get_provider_name(providers[0]) == "TensorrtExecutionProvider":
            if not tensorrt_imported:
                raise ImportError(
                    "tensorrt package is not installed. The package can be install via `pip install tensorrt`."
                )

        self.sess = ort.InferenceSession(onnx_model.SerializeToString(), providers=providers)

        if get_provider_name(providers[0]) == "TensorrtExecutionProvider" and tensorrt_imported:
            assert "TensorrtExecutionProvider" in self.sess.get_providers(), (
                f"unexpected TensorRT compilation failure: TensorrtExecutionProvider not in providers ({self.sess.get_providers()}). "
                "Make sure onnxruntime package gets lazy imported everywhere."
            )

        inputs = self.sess.get_inputs()
        outputs = self.sess.get_outputs()
        self.input_names = [i.name for i in inputs]
        self.output_names = [i.name for i in outputs]

    def __call__(self, *args):
        """
        Make the module callable like torch.nn.Module, while running forward pass with onnxruntime.

        Parameters
        ----------
        args : list of torch.Tensor
            A list of torch.Tensor that are inputs of the model.

        Returns
        -------
        onnx_outputs : list of torch.Tensor
            A list of torch.Tensor that are outputs of the model.
        """
        import torch

        input_dict = {k: args[i].cpu().numpy() for i, k in enumerate(self.input_names)}
        onnx_outputs = self.sess.run(self.output_names, input_dict)
        onnx_outputs = onnx_outputs[:3]
        onnx_outputs = [torch.from_numpy(out) for out in onnx_outputs]
        return onnx_outputs

    def to(self, *args):
        """A dummy function that act as torch.nn.Module.to() function"""

        class DummyModel:
            def eval():
                pass

        return DummyModel
