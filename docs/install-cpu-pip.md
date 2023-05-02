```console
pip install -U pip
pip install -U setuptools wheel

# CPU version of pytorch has smaller footprint - see installation instructions in
# pytorch documentation - https://pytorch.org/get-started/locally/
pip install torchvision~=0.15.1 --force-reinstall --extra-index-url https://download.pytorch.org/whl/cpu

pip install autogluon
```
