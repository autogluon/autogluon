.. note::

  AutoGluon requires `Python <https://www.python.org/downloads/release/python-370/>`_ version 3.6 or 3.7.
  Linux is the only operating system fully supported for now (complete Mac OSX and Windows versions will be available soon).


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

                 pip install --upgrade mxnet
                 pip install autogluon

           .. container:: gpu

              .. code-block:: bash

                 # Here we assume CUDA 10.0 is installed.  You should change the number 
                 # according to your own CUDA version (e.g. mxnet-cu101 for CUDA 10.1).
                 pip install --upgrade mxnet-cu100
                 pip install autogluon

        .. container:: source

           .. container:: cpu

              .. code-block:: bash

                 pip install --upgrade mxnet
                 git clone https://github.com/awslabs/autogluon
                 cd autogluon && python setup.py develop

           .. container:: gpu

              .. code-block:: bash

                 # Here we assume CUDA 10.0 is installed.  You should change the number 
                 # according to your own CUDA version (e.g. mxnet-cu101 for CUDA 10.1).
                 pip install --pre --upgrade mxnet-cu100
                 git clone https://github.com/awslabs/autogluon
                 cd autogluon && python setup.py develop

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

                 pip install --upgrade mxnet
                 pip install autogluon

              .. note::
              
                 AutoGluon is not yet fully functional on Mac OSX. If you encounter MXNet system errors, please use Linux instead.  However, you can currently use AutoGluon for less compute-intensive TabularPrediction tasks on your Mac laptop (but only with hyperparameter_tune = False).

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

                 pip install --upgrade mxnet
                 git clone https://github.com/awslabs/autogluon
                 cd autogluon && python setup.py develop

              .. note::
              
                 AutoGluon is not yet fully functional on Mac OSX. If you encounter MXNet system errors, please use Linux instead.  However, you can currently use AutoGluon for less compute-intensive TabularPrediction tasks on your Mac laptop (but only with hyperparameter_tune = False).

           .. container:: gpu

              .. note::

                 GPU usage is not yet supported on Mac OSX , please use Linux to utilize GPUs in AutoGluon.
