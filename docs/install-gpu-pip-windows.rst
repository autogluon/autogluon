.. code-block:: bash

    pip3 install -U pip
    pip3 install -U setuptools wheel

    # Install the proper version of PyTorch following https://pytorch.org/get-started/locally/
    pip3 install torch==1.12.0+cu113 torchvision==0.13.0+cu113 torchtext==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu113

    pip3 install --pre autogluon
