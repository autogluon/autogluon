.. code-block:: bash

    pip3 install -U pip
    pip3 install -U setuptools wheel

    # Install the proper version of PyTorch following https://pytorch.org/get-started/locally/
    pip3 install torch==1.13.1+cu116 torchvision==0.14.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116

    pip3 install --pre autogluon
