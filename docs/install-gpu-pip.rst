.. code-block:: bash

    pip3 install -U pip
    pip3 install -U setuptools wheel

    # Here we assume CUDA 10.1 is installed.  You should change the number
    # according to your own CUDA version (e.g. mxnet_cu100 for CUDA 10.0).
    pip3 install -U "mxnet_cu101<2.0.0"
    pip3 install --pre autogluon
