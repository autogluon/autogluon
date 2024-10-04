```console
pip install -U pip
pip install -U setuptools wheel
pip install -U uv

# CPU version of pytorch has smaller footprint - see installation instructions in
# pytorch documentation - https://pytorch.org/get-started/locally/
uv pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cpu

uv pip install autogluon
```
