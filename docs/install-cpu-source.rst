.. code-block:: bash

    pip3 install -U pip
    pip3 install -U setuptools wheel
    pip3 install -U "mxnet<2.0.0"

    # CPU version of pytorch has smaller footprint - see installation instructions in
    # pytorch documentation - https://pytorch.org/get-started/locally/
    pip3 install torch==1.10.1+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html

    git clone https://github.com/awslabs/autogluon
    cd autogluon && ./full_install.sh
