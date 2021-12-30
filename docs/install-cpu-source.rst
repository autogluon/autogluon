.. code-block:: bash

    python3 -m pip install -U pip
    python3 -m pip install -U setuptools wheel
    python3 -m pip install -U "mxnet<2.0.0"

    # CPU version of pytorch has smaller footprint - see installation instructions in
    # pytorch documentation - https://pytorch.org/get-started/locally/
    python3 -m pip install torch==1.10.0+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html

    git clone https://github.com/awslabs/autogluon
    cd autogluon && ./full_install.sh
