.. code-block:: bash

    pip3 install -U pip
    pip3 install -U setuptools wheel

    # CPU version of pytorch has smaller footprint - see installation instructions in
    # pytorch documentation - https://pytorch.org/get-started/locally/
    pip3 install torch==1.11.0+cpu torchvision==0.12.0+cpu --extra-index-url https://download.pytorch.org/whl/cpu

    git clone https://github.com/awslabs/autogluon
    cd autogluon && ./full_install.sh
