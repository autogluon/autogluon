Select your preferences and run the install command.

.. role:: title
.. role:: opt
   :class: option
.. role:: act
   :class: active option

.. container:: install

  .. container:: opt-group

     :title:`OS:`
     :opt:`Linux`
     :act:`Mac`

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

                 pip install --pre --upgrade mxnet
                 git clone https://github.com/awslabs/autogluon
                 cd autogluon && python setup.py install --user

           .. container:: cpu

              .. code-block:: bash

                 # Here we assume CUDA 10.0 is installed.  You should change the number 
                 # according to your own CUDA version (e.g. mxnet-cu101 for CUDA 10.1).
                 pip install --pre --upgrade mxnet-cu100
                 git clone https://github.com/awslabs/autogluon
                 cd autogluon && python setup.py install --user

     .. container:: mac

        .. container:: pip

           .. container:: cpu
           
              .. note::
              
                 If you don't have them, you must first install the following packages: 
                 [XCode](https://developer.apple.com/xcode/), [Homebrew](https://brew.sh/), [opencv](https://opencv.org/), [LibOMP](https://formulae.brew.sh/formula/libomp)
                 Once you have Homebrew, the latter two packages can be installed via:

                 .. code-block:: bash

                     brew install libomp opencv

              .. code-block:: bash

                 pip install --upgrade mxnet-osx
                 pip install autogluon

           .. container:: gpu

              .. note::
              
                 If you don't have them, you must first install the following packages: 
                 [XCode](https://developer.apple.com/xcode/), [Homebrew](https://brew.sh/), [opencv](https://opencv.org/), [LibOMP](https://formulae.brew.sh/formula/libomp)
                 Once you have Homebrew, the latter two packages can be installed via:

                 .. code-block:: bash

                     brew install libomp opencv

              Please build MXNet from source to utilize GPU, following detailed instructions from the [MXNet Docs](https://mxnet.apache.org/get_started?version=v1.5.1&platform=macos&language=python&environ=build-from-source&processor=gpu).

              .. code-block:: bash

                 pip install autogluon

        .. container:: source

           .. container:: cpu

              .. note::
              
                 If you don't have them, you must first install the following packages: 
                 [XCode](https://developer.apple.com/xcode/), [Homebrew](https://brew.sh/), [opencv](https://opencv.org/), [LibOMP](https://formulae.brew.sh/formula/libomp)
                 Once you have Homebrew, the latter two packages can be installed via:

                 .. code-block:: bash

                     brew install libomp opencv

              .. code-block:: bash

                 pip install --upgrade mxnet-osx
                 git clone https://github.com/awslabs/autogluon
                 cd autogluon && python setup.py install --user

           .. container:: gpu

              .. note::
              
                 If you don't have them, you must first install the following packages: 
                 [XCode](https://developer.apple.com/xcode/), [Homebrew](https://brew.sh/), [opencv](https://opencv.org/), [LibOMP](https://formulae.brew.sh/formula/libomp)
                 Once you have Homebrew, the latter two packages can be installed via:

                 .. code-block:: bash

                     brew install libomp opencv

              Please build MXNet from source to utilize GPU, following detailed instructions from the [MXNet Docs](https://mxnet.apache.org/get_started?version=v1.5.1&platform=macos&language=python&environ=build-from-source&processor=gpu).

              .. code-block:: bash

                 git clone https://github.com/awslabs/autogluon
                 cd autogluon && python setup.py install --user

  .. note::

     AutoGluon is only supported for [Python](https://www.python.org/downloads/release/python-370/) versions >= 3.6. Make sure [CUDA](https://developer.nvidia.com/cuda-downloads) and [cuDNN](https://developer.nvidia.com/cudnn) are installed if you want to use GPU.

