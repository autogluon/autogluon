.. code-block:: bash

    pip3 install -U pip
    pip3 install -U setuptools wheel

    # CPU version of pytorch has smaller footprint - see installation instructions in
    # pytorch documentation - https://pytorch.org/get-started/locally/
    pip3 install torch==1.13.1+cpu torchvision==0.14.1+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html

    pip3 install autogluon
