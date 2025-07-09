#!/usr/bin/env python
###########################
# This code block is a HACK (!), but is necessary to avoid code duplication. Do NOT alter these lines.
import importlib.util
import os
import platform

from setuptools import setup

filepath = os.path.abspath(os.path.dirname(__file__))
filepath_import = os.path.join(filepath, "..", "core", "src", "autogluon", "core", "_setup_utils.py")
spec = importlib.util.spec_from_file_location("ag_min_dependencies", filepath_import)
ag = importlib.util.module_from_spec(spec)
# Identical to `from autogluon.core import _setup_utils as ag`, but works without `autogluon.core` being installed.
spec.loader.exec_module(ag)
###########################

import sys

def get_cuda_info():
    """Get CUDA version information if available"""
    import subprocess
    import os
    
    # Try nvidia-smi for version info
    try:
        result = subprocess.run(['nvidia-smi'], 
                              capture_output=True, 
                              text=True, 
                              timeout=5)
        if result.returncode == 0:
            # Extract CUDA version from nvidia-smi output
            for line in result.stdout.split('\n'):
                if 'CUDA Version:' in line:
                    return line.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass
    
    # Try nvcc for version info
    try:
        result = subprocess.run(['nvcc', '--version'], 
                              capture_output=True, 
                              text=True, 
                              timeout=5)
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                if 'release' in line.lower():
                    return line.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass
    
    return "CUDA version unknown"

def is_cuda_available():
    """Check if CUDA is available using system detection"""
    import os
    import subprocess
    import sys
    
    print("🔍 Detecting CUDA availability...")
    
    # Check for CUDA environment variables
    if os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH'):
        print("  - CUDA detected via environment variables")
        cuda_info = get_cuda_info()
        print(f"  - {cuda_info}")
        return True
    
    # Check for CUDA libraries in common locations
    cuda_paths = ['/usr/local/cuda', '/opt/cuda', '/usr/cuda']
    for path in cuda_paths:
        if os.path.exists(path):
            print(f"  - CUDA detected via library path: {path}")
            cuda_info = get_cuda_info()
            print(f"  - {cuda_info}")
            return True
    
    # Check for nvidia-smi command
    try:
        result = subprocess.run(['nvidia-smi'], 
                              capture_output=True, 
                              text=True, 
                              timeout=5)
        if result.returncode == 0:
            print("  - CUDA detected via nvidia-smi")
            cuda_info = get_cuda_info()
            print(f"  - {cuda_info}")
            return True
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass
    
    # Check for CUDA libraries in system paths
    try:
        if sys.platform.startswith('linux'):
            # Check for libcuda.so
            result = subprocess.run(['ldconfig', '-p'], 
                                  capture_output=True, 
                                  text=True)
            if result.returncode == 0 and 'libcuda.so' in result.stdout:
                print("  - CUDA detected via system libraries")
                cuda_info = get_cuda_info()
                print(f"  - {cuda_info}")
                return True
        elif sys.platform.startswith('win'):
            # Check for CUDA DLLs on Windows
            cuda_dlls = ['cuda.dll', 'cudart.dll']
            for dll in cuda_dlls:
                try:
                    subprocess.run(['where', dll], 
                                 capture_output=True, 
                                 check=True)
                    print(f"  - CUDA detected via DLL: {dll}")
                    cuda_info = get_cuda_info()
                    print(f"  - {cuda_info}")
                    return True
                except subprocess.CalledProcessError:
                    continue
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass
    
    print("  - No CUDA detected")
    return False

version = ag.load_version_file()
version = ag.update_version(version)

submodule = "tabular"
install_requires = [
    # version ranges added in ag.get_dependency_version_ranges()
    "numpy",  # version range defined in `core/_setup_utils.py`
    "scipy",  # version range defined in `core/_setup_utils.py`
    "pandas",  # version range defined in `core/_setup_utils.py`
    "scikit-learn",  # version range defined in `core/_setup_utils.py`
    "networkx",  # version range defined in `core/_setup_utils.py`
    f"{ag.PACKAGE_NAME}.core=={version}",
    f"{ag.PACKAGE_NAME}.features=={version}",
]

extras_require = {
    "lightgbm": [
        "lightgbm>=4.0,<4.7",  # <{N+1} upper cap, where N is the latest released minor version
    ],
    "catboost": [
        "numpy>=1.25,<2.3.0",
        "catboost>=1.2,<1.3",
    ],
    "xgboost": [
        "xgboost>=2.0,<3.1",  # <{N+1} upper cap, where N is the latest released minor version
    ],
    "fastai": [
        "spacy<3.9",
        "torch",  # version range defined in `core/_setup_utils.py`
        "fastai>=2.3.1,<2.9",  # <{N+1} upper cap, where N is the latest released minor version
    ],
    "tabm": [
        "torch",  # version range defined in `core/_setup_utils.py`
    ],
    "tabpfn": [
        # versions below 0.1.11 are yanked, not compatible with >=2.0.0 yet
        "tabpfn>=0.1.11,<2.0",  # after v2 compatibility is ensured, should be <{N+1} upper cap, where N is the latest released minor version
    ],
    "tabpfnmix": [
        "torch",  # version range defined in `core/_setup_utils.py`
        "huggingface_hub[torch]",  # Only needed for HuggingFace downloads, currently uncapped to minimize future conflicts.
        "einops>=0.7,<0.9",
    ],
    "mitra": [
        "loguru",
        "einx",
        "omegaconf",
        "transformers",
        # "flash-attn>2.5.0,<2.8",
        # "flash-attn==2.6.3",
    ],
    "ray": [
        f"{ag.PACKAGE_NAME}.core[all]=={version}",
    ],
    "skex": [
        "scikit-learn-intelex>=2024.0,<2025.5",  # <{N+1} upper cap, where N is the latest released minor version
    ],
    "imodels": [
        "imodels>=1.3.10,<2.1.0",  # 1.3.8/1.3.9 either remove/renamed attribute `complexity_` causing failures. https://github.com/csinva/imodels/issues/147
    ],
}

cuda_detected = is_cuda_available()
if cuda_detected:
    print("CUDA detected - including flash-attn in mitra extras")
    extras_require["mitra"].append("flash-attn==2.6.3")
else:
    print("CUDA not detected - skipping flash-attn installation")
extras_require["cuda"] = [
    "flash-attn==2.6.3",
]

is_aarch64 = platform.machine() == "aarch64"
is_darwin = sys.platform == "darwin"

if is_darwin or is_aarch64:
    # For macOS or aarch64, only use CPU version
    extras_require["skl2onnx"] = [
        "onnx>=1.13.0,<1.16.2;platform_system=='Windows'",  # cap at 1.16.1 for issue https://github.com/onnx/onnx/issues/6267
        "onnx>=1.13.0,<1.18.0;platform_system!='Windows'",
        "skl2onnx>=1.15.0,<1.18.0",
        # For macOS, there isn't a onnxruntime-gpu package installed with skl2onnx.
        # Therefore, we install onnxruntime explicitly here just for macOS.
        "onnxruntime>=1.17.0,<1.20.0",
    ]
else:
    # For other platforms, include both CPU and GPU versions
    extras_require["skl2onnx"] = [
        "onnx>=1.13.0,<1.16.2;platform_system=='Windows'",  # cap at 1.16.1 for issue https://github.com/onnx/onnx/issues/6267
        "onnx>=1.13.0,<1.18.0;platform_system!='Windows'",
        "skl2onnx>=1.15.0,<1.18.0",
        "onnxruntime>=1.17.0,<1.20.0",  # install for gpu system due to https://github.com/autogluon/autogluon/issues/3804
        "onnxruntime-gpu>=1.17.0,<1.20.0",
    ]

# TODO: v1.0: Rename `all` to `core`, make `all` contain everything.
all_requires = []
# TODO: Consider adding 'skex' to 'all'
for extra_package in ["lightgbm", "catboost", "xgboost", "fastai", "tabm", "tabpfnmix", "ray"]:
    all_requires += extras_require[extra_package]
all_requires = list(set(all_requires))
extras_require["all"] = all_requires


test_requires = []
for test_package in ["tabpfnmix", "mitra", "imodels", "skl2onnx"]:
    test_requires += extras_require[test_package]
extras_require["tests"] = test_requires
install_requires = ag.get_dependency_version_ranges(install_requires)
extras_require = {key: ag.get_dependency_version_ranges(value) for key, value in extras_require.items()}

if __name__ == "__main__":
    ag.create_version_file(version=version, submodule=submodule)
    setup_args = ag.default_setup_args(version=version, submodule=submodule)
    setup(
        install_requires=install_requires,
        extras_require=extras_require,
        **setup_args,
    )
