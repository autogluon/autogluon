.. code-block:: bash

    python3 -m pip install -U pip
    python3 -m pip install -U setuptools wheel

    # Here we assume CUDA 10.1 is installed.  You should change the number
    # according to your own CUDA version (e.g. mxnet_cu102 for CUDA 10.2).
    python3 -m pip install -U "mxnet_cu101<2.0.0"
    git clone https://github.com/awslabs/autogluon
    cd autogluon && ./full_install.sh
