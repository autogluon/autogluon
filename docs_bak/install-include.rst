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
     :opt:`CUDA`

     .. raw:: html

        <div class="mdl-tooltip" data-mdl-for="cpu">Build-in backend for CPU.</div>
        <div class="mdl-tooltip" data-mdl-for="cuda">Required to run on Nvidia GPUs.</div>

  .. admonition:: Command:

     .. container:: linux

        .. container:: pip

           .. container:: cpu

              .. code-block:: bash

                 pip install --upgrade mxnet
                 pip install autogluon

           .. container:: cuda

              .. code-block:: bash

                 # Here we assume CUDA 10.0 is installed. You can change the number
                 # according to your own CUDA version.
                 pip install --upgrade mxnet-cu100
                 pip install autogluon

        .. container:: source

           .. container:: cpu

              .. code-block:: bash

                 pip install --pre --upgrade mxnet
                 git clone https://github.com/awslabs/autogluon
                 cd autogluon && python setup.py install --user

           .. container:: cuda

              .. code-block:: bash

                 # Here we assume CUDA 10.0 is installed. You can change the number
                 # according to your own CUDA version.
                 pip install --pre --upgrade mxnet-cu100
                 git clone https://github.com/awslabs/autogluon
                 cd autogluon && python setup.py install --user

     .. container:: mac

        .. container:: pip

           .. container:: cpu

              .. code-block:: bash

                 pip install --upgrade mxnet-osx
                 pip install autogluon

           .. container:: cuda

              Please build MXNet from source manually, details in `MXNet Docs<https://mxnet.apache.org/get_started?version=v1.5.1&platform=macos&language=python&environ=build-from-source&processor=gpu>`_.

              .. code-block:: bash

                 pip install autogluon

        .. container:: source

           .. container:: cpu

              .. code-block:: bash

                 pip install --upgrade mxnet-osx
                 git clone https://github.com/awslabs/autogluon
                 cd autogluon && python setup.py install --user

           .. container:: cuda

              Please build MXNet from source manually, details in `MXNet Docs<https://mxnet.apache.org/get_started?version=v1.5.1&platform=macos&language=python&environ=build-from-source&processor=gpu>`_.

              .. code-block:: bash

                 git clone https://github.com/awslabs/autogluon
                 cd autogluon && python setup.py install --user

