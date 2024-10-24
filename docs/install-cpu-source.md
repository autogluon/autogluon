```console
pip install -U pip
pip install -U setuptools wheel
pip install -U uv

# CPU version of pytorch has smaller footprint - see installation instructions in
# pytorch documentation - https://pytorch.org/get-started/locally/
uv pip install torch==2.4.1+cpu torchvision==0.19.1+cpu --index-url https://download.pytorch.org/whl/cpu

git clone https://github.com/autogluon/autogluon
cd autogluon && ./full_install.sh
```
