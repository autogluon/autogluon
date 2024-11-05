```console
# Install UV package installer (faster than pip)
pip install -U uv

# GPU version of pytorch with CUDA support
python -m uv pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu118

# Install AutoGluon with GPU support
python -m uv pip install autogluon
```