.. code-block:: bash

    pip3 install -U pip
    pip3 install -U setuptools wheel

    # Note: GPU MXNet is not supported on Windows, so we don't install MXNet.
    git clone https://github.com/awslabs/autogluon
    cd autogluon && ./full_install.sh
