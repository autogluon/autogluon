Installation
============

* Install from source

::

    git clone https://github.com/awslabs/auto-ml-with-gluon.git && cd auto-ml-with-gluon
    python setup.py install

.. note::

    Before installing from source, please have the MXNet and python installed.
    We use CUDA 10.0 and python 3.7 on Linux as an example.

::

    wget https://repo.anaconda.com/archive/Anaconda3-2019.07-Linux-x86_64.sh
    sh Anaconda3-2019.07-Linux-x86_64.sh
    pip install mxnet-cu100


* Install using pip

::

    pip install http://autogluon-hackathon.s3-website-us-west-2.amazonaws.com/dist/autogluon-0.0.1+9c8fe01-py3-none-any.whl

.. note::

    The whl would be updated soon.
