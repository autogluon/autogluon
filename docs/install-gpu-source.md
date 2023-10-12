```console
pip install -U pip
pip install -U setuptools wheel

# Install the proper version of PyTorch following https://pytorch.org/get-started/locally/
pip install torchvision~=0.15.1 --force-reinstall --extra-index-url https://download.pytorch.org/whl/cu118

git clone https://github.com/autogluon/autogluon
cd autogluon && ./full_install.sh
```
