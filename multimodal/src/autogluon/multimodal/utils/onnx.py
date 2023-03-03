import logging
from typing import Dict, List, Optional, Tuple, Union

from torch import tensor

from ..constants import AUTOMM, FEATURE_EXTRACTION, MULTICLASS

logger = logging.getLogger(__name__)

try:
    import tensorrt  # Unused but required by TensorrtExecutionProvider
except:
    logger.warning(
        "Failed to import tensorrt package. "
        "onnxruntime would fallback to CUDAExecutionProvider instead of using TensorrtExecutionProvider."
    )


def onnx_get_dynamic_axes(input_keys: List[str]):
    dynamic_axes = {}
    for k in input_keys:
        if "token_ids" in k or "segment_ids" in k:
            dynamic_axes[k] = {0: "batch_size", 1: "seq_length"}
        elif "valid_length" in k or k.startswith("numerical") or k.startswith("timm_image"):
            dynamic_axes[k] = {0: "batch_size"}

    return dynamic_axes


class OnnxModule(object):
    """
    OnnxModule is as a replacement of torch.nn.Module for running forward pass with onnxruntime.

    The module can be generated with MultiModalPredictor.export_tensorrt(),
    so that we can predict with TensorRT by simply replacing predictor._model with OnnxModule.
    """

    def __init__(self, model, providers: Optional[Union[dict, List[str]]] = None):
        """
        Parameters
        ----------
        model : onnx.ModelProto
            The onnx model that need to be executed in onnxruntime.
        providers : dict or str, default=None
            A list of execution providers for model prediction in onnxruntime.
        """
        import onnxruntime as ort

        if providers == None:
            providers = [
                (
                    "TensorrtExecutionProvider",
                    {
                        "device_id": 0,
                        "trt_max_workspace_size": 2147483648,
                        "trt_fp16_enable": True,
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
        self.sess = ort.InferenceSession(model.SerializeToString(), providers=providers)
        inputs = self.sess.get_inputs()
        outputs = self.sess.get_outputs()
        self.input_names = [i.name for i in inputs]
        self.output_names = [i.name for i in outputs]

    def __call__(self, *args):
        """
        Make the module callable like torch.nn.Module, while runnning forward pass with onnxruntime.

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
