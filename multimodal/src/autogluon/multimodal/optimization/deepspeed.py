# Slightly adapted file of pytorch_lightning.strategies.deepspeed to not init deepspeed using pytorch-lightning's deepspeed inference workaround as this has higher memory requirements and can result in OOM during inference. Instead fallback to initialization using `deepspeed.initialize`.
# TODO: Support deepspeed_inference, custom kernels, and quantization for fast inference.

# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
from typing import Any, Dict, Generator, List, Mapping, Optional, Tuple, Union, cast

import pytorch_lightning as pl
import torch
from pytorch_lightning.accelerators.cuda import CUDAAccelerator
from pytorch_lightning.overrides.base import _LightningModuleWrapperBase, _LightningPrecisionModuleWrapperBase
from pytorch_lightning.plugins.environments.cluster_environment import ClusterEnvironment
from pytorch_lightning.plugins.precision import PrecisionPlugin
from pytorch_lightning.strategies import DeepSpeedStrategy
from pytorch_lightning.utilities import GradClipAlgorithmType
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.model_helpers import is_overridden
from pytorch_lightning.utilities.rank_zero import rank_zero_warn
from pytorch_lightning.utilities.types import _PATH


class CustomDeepSpeedStrategy(DeepSpeedStrategy):
    """
    Provides capabilities to run training using the DeepSpeed library, with training optimizations for large
        billion parameter models. `For more information: https://pytorch-
        lightning.readthedocs.io/en/stable/advanced/model_parallel.html#deepspeed`.

        .. warning:: ``DeepSpeedStrategy`` is in beta and subject to change.

        Defaults have been set to enable ZeRO-Offload and some have been taken from the link below.
        These defaults have been set generally, but may require tuning for optimum performance based on your model size.
        `For more information: https://www.deepspeed.ai/docs/config-json/#zero-optimizations-for-fp16-training`.
    """

    def __init__(
        self,
        accelerator: Optional["pl.accelerators.accelerator.Accelerator"] = None,
        zero_optimization: bool = True,
        stage: int = 2,
        remote_device: str = "cpu",
        offload_optimizer: bool = False,
        offload_parameters: bool = False,
        offload_params_device: str = "cpu",
        nvme_path: str = "/local_nvme",
        params_buffer_count: int = 5,
        params_buffer_size: int = 100_000_000,
        max_in_cpu: int = 1_000_000_000,
        offload_optimizer_device: str = "cpu",
        optimizer_buffer_count: int = 4,
        block_size: int = 1048576,
        queue_depth: int = 8,
        single_submit: bool = False,
        overlap_events: bool = True,
        thread_count: int = 1,
        pin_memory: bool = False,
        sub_group_size: int = 1_000_000_000_000,
        contiguous_gradients: bool = True,
        overlap_comm: bool = True,
        allgather_partitions: bool = True,
        reduce_scatter: bool = True,
        allgather_bucket_size: int = 200_000_000,
        reduce_bucket_size: int = 200_000_000,
        zero_allow_untested_optimizer: bool = True,
        logging_batch_size_per_gpu: Union[str, int] = "auto",
        config: Optional[Union[_PATH, Dict[str, Any]]] = None,
        logging_level: int = logging.WARN,
        parallel_devices: Optional[List[torch.device]] = None,
        cluster_environment: Optional[ClusterEnvironment] = None,
        loss_scale: float = 0,
        initial_scale_power: int = 16,
        loss_scale_window: int = 1000,
        hysteresis: int = 2,
        min_loss_scale: int = 1,
        partition_activations: bool = False,
        cpu_checkpointing: bool = False,
        contiguous_memory_optimization: bool = False,
        synchronize_checkpoint_boundary: bool = False,
        load_full_weights: bool = False,
        precision_plugin: Optional[PrecisionPlugin] = None,
        process_group_backend: Optional[str] = None,
    ) -> None:
        """
        Parameters
        ----------
        zero_optimization
            Enable ZeRO optimization. This is only compatible with precision=16.

        stage
            Different stages of the ZeRO Optimizer. 0 is disabled,
            1 is optimizer state partitioning, 2 is optimizer+gradient state partitioning,
            3 is optimizer+gradient_parameter partitioning using the infinity engine.

        remote_device
            Device to instantiate the model on initially (``cpu`` or ``nvme``).

        offload_optimizer
            Enable offloading optimizer memory and computation to CPU or NVMe
            based on ``offload_optimizer_device``.

        offload_parameters
            When using ZeRO Stage 3, Enable offloading parameter memory and computation
            to CPU or NVMe based on ``offload_params_device``.

        offload_params_device
            When offloading parameters choose the device to offload to, ``cpu`` or ``nvme``.

        offload_optimizer_device
            When offloading optimizer state choose the device to offload to,
            ``cpu`` or ``nvme``.

        params_buffer_count
            Number of buffers in buffer pool for
            parameter offloading when ``offload_params_device`` is ``nvme``.

        params_buffer_size
            Size of buffers in buffer pool for parameter offloading
            when ``offload_params_device`` is ``nvme``.

        max_in_cpu
            Number of parameter elements to maintain in CPU memory when offloading to NVMe is enabled.

        nvme_path
            Filesystem path for NVMe device for optimizer/parameter state offloading.

        optimizer_buffer_count
            Number of buffers in buffer pool for optimizer state offloading
            when ``offload_optimizer_device`` is set to to ``nvme``.
            This should be at least the number of states maintained per parameter by the optimizer.
            For example, Adam optimizer has 4 states (parameter, gradient, momentum, and variance).

        block_size
            When using NVMe Offloading, the I/O block size in bytes.

        queue_depth
            When using NVMe Offloading, the I/O queue depth.

        single_submit
            When using NVMe Offloading,
            submit requests to storage device as multiple individual requests,
            as opposed to one block of requests.

        overlap_events
            When using NVMe Offloading,
            submit requests to storage device in an overlapped fashion
            without waiting for completion of earlier requests.

        thread_count
            When using NVMe Offloading,
            Intra-request parallelism for each read/write submitted by a user thread.

        pin_memory
            When using ZeRO stage 3, pin optimizer state memory on CPU.
            This could boost throughput at the cost of extra memory overhead.

        sub_group_size
            When using ZeRO stage 3, defines the number of parameters
            within a sub group to offload at a time.
            Smaller numbers require more communication, but improve memory efficiency.

        contiguous_gradients
            Copies gradients to a continuous buffer as they are produced.
            Avoids memory fragmentation during backwards. Useful when training large models.

        overlap_comm
            Overlap the reduction (synchronization) of gradients with the backwards computation.
            This is a speed optimization when training across multiple GPUs/machines.

        allgather_partitions: All gather updated parameters at the end of training step,
            instead of using a series of broadcast collectives.

        reduce_scatter: Use reduce/scatter instead of allreduce to average gradients.

        allgather_bucket_size
            Number of elements to allgather at once.
            Used to limit the memory required for larger model sizes, with a tradeoff with speed.

        reduce_bucket_size
            Number of elements to reduce at once.
            Used to limit the memory required for larger model sizes, with a tradeoff with speed.

        zero_allow_untested_optimizer
            Allow untested optimizers to be used with ZeRO. Currently only Adam is a
            DeepSpeed supported optimizer when using ZeRO.

        logging_batch_size_per_gpu
            Config used in DeepSpeed to calculate verbose timing for logging
            on a per sample per second basis (only displayed if logging=logging.INFO).
            If set to "auto", the plugin tries to infer this from
            the train DataLoader's BatchSampler, else defaults to 1.
            To obtain accurate logs when using datasets that do not support batch samplers,
            set this to the actual per gpu batch size (trainer.batch_size).

        config
            Pass in a deepspeed formatted config dict,
            or path to a deepspeed config: https://www.deepspeed.ai/docs/config-json.
            All defaults will be ignored if a config is passed in.

        logging_level
            Set logging level for deepspeed.

        loss_scale
            Loss scaling value for FP16 training.
            0.0 results in dynamic loss scaling, otherwise static.

        initial_scale_power
            Power of the initial dynamic loss scale value. Loss scale is computed
            by ``2^initial_scale_power``.

        loss_scale_window
            Window in which to raise/lower the dynamic FP16 loss scaling value.

        hysteresis
            FP16 Delay shift in Dynamic Loss scaling.

        min_loss_scale
            The minimum FP16 dynamic loss scaling value.

        partition_activations
            Enables partition activation when used with ZeRO stage 3 and model parallelism.
            Still requires you to wrap your forward functions in deepspeed.checkpointing.checkpoint.
            See `deepspeed tutorial
            <https://www.deepspeed.ai/tutorials/megatron/#deepspeed-activation-checkpoints-optional>`_.

        cpu_checkpointing
            Offloads partitioned activations to CPU if ``partition_activations`` is enabled.

        contiguous_memory_optimization
            Copies partitioned activations so that they are contiguous in memory.
            Not supported by all models.

        synchronize_checkpoint_boundary
            Insert :func:`torch.cuda.synchronize` at each checkpoint boundary.

        load_full_weights
            True when loading a single checkpoint file containing the model state dict
            when using ZeRO Stage 3. This differs from the DeepSpeed checkpoint which contains shards
            per worker.
        """
        super().__init__(
            accelerator=accelerator,
            parallel_devices=parallel_devices,
            cluster_environment=cluster_environment,
            precision_plugin=precision_plugin,
            process_group_backend=process_group_backend,
            zero_optimization=zero_optimization,
            stage=stage,
            remote_device=remote_device,
            offload_optimizer=offload_optimizer,
            offload_parameters=offload_parameters,
            params_buffer_count=params_buffer_count,
            params_buffer_size=params_buffer_size,
            max_in_cpu=max_in_cpu,
            offload_optimizer_device=offload_optimizer_device,
            optimizer_buffer_count=optimizer_buffer_count,
            block_size=block_size,
            queue_depth=queue_depth,
            single_submit=single_submit,
            overlap_events=overlap_events,
            thread_count=thread_count,
            pin_memory=pin_memory,
            sub_group_size=sub_group_size,
            contiguous_gradients=contiguous_gradients,
            overlap_comm=overlap_comm,
            allgather_partitions=allgather_partitions,
            reduce_scatter=reduce_scatter,
            allgather_bucket_size=allgather_bucket_size,
            reduce_bucket_size=reduce_bucket_size,
            zero_allow_untested_optimizer=zero_allow_untested_optimizer,
            logging_batch_size_per_gpu=logging_batch_size_per_gpu,
            config=config,
            logging_level=logging_level,
            loss_scale=loss_scale,
            initial_scale_power=initial_scale_power,
            loss_scale_window=loss_scale_window,
            hysteresis=hysteresis,
            min_loss_scale=min_loss_scale,
            partition_activations=partition_activations,
            cpu_checkpointing=cpu_checkpointing,
            contiguous_memory_optimization=contiguous_memory_optimization,
            synchronize_checkpoint_boundary=synchronize_checkpoint_boundary,
            load_full_weights=load_full_weights,
        )

    def init_deepspeed(self) -> None:
        assert self.lightning_module is not None
        # deepspeed handles gradient clipping internally
        if is_overridden("configure_gradient_clipping", self.lightning_module, pl.LightningModule):
            rank_zero_warn(
                "Since DeepSpeed handles gradient clipping internally, the default"
                " `LightningModule.configure_gradient_clipping` implementation will not actually clip gradients."
                " The hook will still be called. Consider setting"
                " `Trainer(gradient_clip_val=..., gradient_clip_algorithm='norm')`"
                " which will use the internal mechanism."
            )

        if self.lightning_module.trainer.gradient_clip_algorithm == GradClipAlgorithmType.VALUE:
            raise MisconfigurationException("DeepSpeed does not support clipping gradients by value.")

        if not isinstance(self.accelerator, CUDAAccelerator):
            raise MisconfigurationException(
                f"DeepSpeed strategy is only supported on GPU but `{self.accelerator.__class__.__name__}` is used."
            )

        accumulation_scheduler = self.lightning_module.trainer.accumulation_scheduler

        if accumulation_scheduler.epochs != [0]:
            raise MisconfigurationException(
                "DeepSpeed currently does not support different `accumulate_grad_batches` at different epochs."
            )

        assert isinstance(self.model, (pl.LightningModule, _LightningPrecisionModuleWrapperBase))
        model = _LightningModuleWrapperBase(pl_module=self.model)

        self._initialize_deepspeed_train(model)
