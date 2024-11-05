```console
# Install UV package installer (faster than pip)
pip install -U uv

# CPU version of pytorch
python -m uv pip install torch==2.4.1+cpu torchvision==0.19.1+cpu --index-url https://download.pytorch.org/whl/cpu

# Install AutoGluon
python -m uv pip install autogluon
```