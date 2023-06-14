```console
pip install -U pip
pip install -U setuptools wheel

# Install the proper version of PyTorch following https://pytorch.org/get-started/locally/
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117

git clone https://github.com/autogluon/autogluon
cd autogluon && ./full_install.sh
```
