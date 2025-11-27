from __future__ import annotations

import logging
import os
import sys
import warnings
from dataclasses import dataclass, fields
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np


@dataclass
class GlobalContextSnapshot:
    """
    Snapshot of selected process-global state to verify that an operation
    (e.g., TabularPredictor.fit) does not leak global config changes.

    Use:
    ----
    before = GlobalContextSnapshot.capture()
    ... run code under test ...
    after = GlobalContextSnapshot.capture()
    before.assert_unchanged(after)
    """

    # Torch-related
    has_torch: bool
    torch_num_threads: Optional[int]
    torch_num_interop_threads: Optional[int]
    torch_default_dtype: Optional[str]
    torch_cudnn_benchmark: Optional[bool]
    torch_cudnn_deterministic: Optional[bool]
    torch_cudnn_enabled: Optional[bool]
    torch_cuda_is_available: Optional[bool]
    torch_matmul_allow_tf32: Optional[bool]
    torch_cudnn_allow_tf32: Optional[bool]

    # NumPy config
    numpy_error_state: Mapping[str, str]
    numpy_print_options: Mapping[str, Any]

    # Environment / filesystem
    cwd: str
    env_snapshot: Dict[str, Optional[str]]

    # Logging
    logging_root_level: int
    logging_root_handler_types: Tuple[str, ...]

    # Warnings
    warnings_filters: List[Any]

    # Import system
    sys_path: List[str]
    sys_meta_path_types: Tuple[str, ...]

    # Which env vars we care about
    _ENV_KEYS: Tuple[str, ...] = (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    )

    @classmethod
    def capture(cls) -> "GlobalContextSnapshot":
        """Capture the current global context."""
        # Torch
        try:
            import torch  # type: ignore[import-not-found]
        except ImportError:
            has_torch = False
            torch_num_threads = None
            torch_num_interop_threads = None
            torch_default_dtype = None
            torch_cudnn_benchmark = None
            torch_cudnn_deterministic = None
            torch_cudnn_enabled = None
            torch_cuda_is_available = None
            torch_matmul_allow_tf32 = None
            torch_cudnn_allow_tf32 = None
        else:
            has_torch = True

            # Basic thread / dtype / cuda flags
            torch_num_threads = torch.get_num_threads()
            torch_num_interop_threads = (
                torch.get_num_interop_threads()
                if hasattr(torch, "get_num_interop_threads")
                else None
            )
            torch_default_dtype = str(torch.get_default_dtype())
            torch_cuda_is_available = torch.cuda.is_available()

            # Backends: cuDNN / deterministic / TF32
            torch_cudnn_benchmark = torch.backends.cudnn.benchmark
            torch_cudnn_deterministic = torch.backends.cudnn.deterministic
            torch_cudnn_enabled = torch.backends.cudnn.enabled

            if hasattr(torch.backends, "cuda") and hasattr(
                torch.backends.cuda, "matmul"
            ):
                torch_matmul_allow_tf32 = torch.backends.cuda.matmul.allow_tf32
            else:
                torch_matmul_allow_tf32 = None

            torch_cudnn_allow_tf32 = (
                torch.backends.cudnn.allow_tf32
                if hasattr(torch.backends.cudnn, "allow_tf32")
                else None
            )

        # NumPy
        numpy_error_state = dict(np.geterr())
        numpy_print_options = dict(np.get_printoptions())

        # Env / cwd
        cwd = os.getcwd()
        env_snapshot = {k: os.environ.get(k) for k in cls._ENV_KEYS}

        # Logging
        root_logger = logging.getLogger()
        logging_root_level = root_logger.level
        logging_root_handler_types = tuple(
            f"{h.__class__.__module__}.{h.__class__.__name__}"
            for h in root_logger.handlers
        )

        # Warnings
        warnings_filters = list(warnings.filters)

        # Import system
        sys_path = list(sys.path)
        sys_meta_path_types = tuple(
            f"{mp.__class__.__module__}.{mp.__class__.__name__}" for mp in sys.meta_path
        )

        return cls(
            has_torch=has_torch,
            torch_num_threads=torch_num_threads,
            torch_num_interop_threads=torch_num_interop_threads,
            torch_default_dtype=torch_default_dtype,
            torch_cudnn_benchmark=torch_cudnn_benchmark,
            torch_cudnn_deterministic=torch_cudnn_deterministic,
            torch_cudnn_enabled=torch_cudnn_enabled,
            torch_cuda_is_available=torch_cuda_is_available,
            torch_matmul_allow_tf32=torch_matmul_allow_tf32,
            torch_cudnn_allow_tf32=torch_cudnn_allow_tf32,
            numpy_error_state=numpy_error_state,
            numpy_print_options=numpy_print_options,
            cwd=cwd,
            env_snapshot=env_snapshot,
            logging_root_level=logging_root_level,
            logging_root_handler_types=logging_root_handler_types,
            warnings_filters=warnings_filters,
            sys_path=sys_path,
            sys_meta_path_types=sys_meta_path_types,
        )

    def assert_unchanged(self, other: "GlobalContextSnapshot") -> None:
        """
        Assert that `other` matches this snapshot.

        Raises
        ------
        AssertionError
            If any tracked field differs between the two snapshots.
        """
        diffs: list[str] = []

        # Fields that are either helpers or too noisy / dynamic to assert on
        skip_fields = {"_ENV_KEYS", "sys_meta_path_types", "warnings_filters", "sys_path"}

        for f in fields(self):
            if f.name in skip_fields:
                continue
            before_val = getattr(self, f.name)
            after_val = getattr(other, f.name)
            if before_val != after_val:
                diffs.append(
                    f"- {f.name} changed:\n"
                    f"    before={before_val!r}\n"
                    f"    after ={after_val!r}"
                )

        # --- Softer sys.path check: before must be a subsequence of after ----
        def _is_subsequence(sub: list[str], full: list[str]) -> bool:
            it = iter(full)
            for wanted in sub:
                for candidate in it:
                    if candidate == wanted:
                        break
                else:
                    # Exhausted 'full' without finding 'wanted'
                    return False
            return True

        if self.sys_path != other.sys_path:
            if not _is_subsequence(self.sys_path, other.sys_path):
                diffs.append(
                    "- sys_path changed (not just extra entries inserted):\n"
                    f"    before={self.sys_path!r}\n"
                    f"    after ={other.sys_path!r}"
                )

        if diffs:
            msg = "Global context changed across operation:\n" + "\n".join(diffs)
            raise AssertionError(msg)
