.. include:: install-warning.rst

.. note::

  AutoGluon requires `Python <https://www.python.org/downloads/release/python-370/>`_ version 3.6, 3.7, or 3.8.
  Linux and Mac are fully supported, we recommend Windows users to run AutoGluon through `Docker <https://hub.docker.com/r/autogluon/autogluon>`_. 
  For troubleshooting the installation process, you can check the `Installation FAQ <install.html#installation-faq>`_.


Select your preferences below and run the corresponding install commands:

.. role:: title
.. role:: opt
   :class: option
.. role:: act
   :class: active option

.. container:: install

  .. container:: opt-group

     :title:`OS:`
     :act:`Linux`
     :opt:`Mac`

     .. raw:: html

        <div class="mdl-tooltip" data-mdl-for="linux">Linux.</div>
        <div class="mdl-tooltip" data-mdl-for="mac">Mac OSX.</div>

  .. container:: opt-group

     :title:`Version:`
     :act:`PIP`
     :opt:`Source`

     .. raw:: html

        <div class="mdl-tooltip" data-mdl-for="pip">PIP Release.</div>
        <div class="mdl-tooltip" data-mdl-for="source">Install AutoGluon from source.</div>


  .. container:: opt-group

     :title:`Backend:`
     :act:`CPU`
     :opt:`GPU`

     .. raw:: html

        <div class="mdl-tooltip" data-mdl-for="cpu">Build-in backend for CPU.</div>
        <div class="mdl-tooltip" data-mdl-for="gpu">Required to run on Nvidia GPUs.</div>

  .. admonition:: Command:

     .. container:: linux

        .. container:: pip

           .. container:: cpu

              .. code-block:: bash

                 python3 -m pip install -U pip
                 python3 -m pip install -U setuptools wheel
                 python3 -m pip install -U "mxnet<2.0.0"  # Optional
                 python3 -m pip install --pre autogluon

           .. container:: gpu

              .. code-block:: bash

                 python3 -m pip install -U pip
                 python3 -m pip install -U setuptools wheel

                 # Here we assume CUDA 10.1 is installed.  You should change the number
                 # according to your own CUDA version (e.g. mxnet_cu100 for CUDA 10.0).
                 python3 -m pip install -U "mxnet_cu101<2.0.0"  # Optional
                 python3 -m pip install --pre autogluon

        .. container:: source

           .. container:: cpu

              .. code-block:: bash

                 python3 -m pip install -U pip
                 python3 -m pip install -U setuptools wheel
                 python3 -m pip install -U "mxnet<2.0.0"  # Optional
                 git clone https://github.com/awslabs/autogluon
                 cd autogluon && ./full_install.sh

           .. container:: gpu

              .. code-block:: bash

                 python3 -m pip install -U pip
                 python3 -m pip install -U setuptools wheel

                 # Here we assume CUDA 10.1 is installed.  You should change the number
                 # according to your own CUDA version (e.g. mxnet_cu102 for CUDA 10.2).
                 python3 -m pip install -U "mxnet_cu101<2.0.0"  # Optional
                 git clone https://github.com/awslabs/autogluon
                 cd autogluon && ./full_install.sh

     .. container:: mac

        .. container:: pip

           .. container:: cpu

              .. note::

                 If you don't have them, please first install:
                 `XCode <https://developer.apple.com/xcode/>`_, `Homebrew <https://brew.sh>`_, `LibOMP <https://formulae.brew.sh/formula/libomp>`_.
                 Once you have Homebrew, LibOMP can be installed via:

                 .. code-block:: bash

                     brew install libomp

              .. code-block:: bash

                 python3 -m pip install -U pip
                 python3 -m pip install -U setuptools wheel
                 python3 -m pip install -U "mxnet<2.0.0"
                 python3 -m pip install --pre autogluon

           .. container:: gpu

              .. note::

                 GPU usage is not yet supported on Mac OSX, please use Linux to utilize GPUs in AutoGluon.

        .. container:: source

           .. container:: cpu

              .. note::

                 If you don't have them, please first install:
                 `XCode <https://developer.apple.com/xcode/>`_, `Homebrew <https://brew.sh>`_, `LibOMP <https://formulae.brew.sh/formula/libomp>`_.
                 Once you have Homebrew, LibOMP can be installed via:

                 .. code-block:: bash

                     brew install libomp

              .. code-block:: bash

                 python3 -m pip install -U pip
                 python3 -m pip install -U setuptools wheel
                 python3 -m pip install -U "mxnet<2.0.0"
                 git clone https://github.com/awslabs/autogluon
                 cd autogluon && ./full_install.sh

           .. container:: gpu

              .. note::

                 GPU usage is not yet supported on Mac OSX , please use Linux to utilize GPUs in AutoGluon.
