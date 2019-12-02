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
                 pip install https://autogluon.s3.amazonaws.com/dist/autogluon-0.0.1-py3-none-any.whl

           .. container:: cuda

              .. code-block:: bash

                 # Here we assume CUDA 10.0 is installed. You can change the number
                 # according to your own CUDA version.
                 pip install --upgrade mxnet-cu100
                 pip install https://autogluon.s3.amazonaws.com/dist/autogluon-0.0.1-py3-none-any.whl

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

                 pip install https://autogluon.s3.amazonaws.com/dist/mxnet-1.6.0b20191125-cp37-cp37m-macosx_10_13_x86_64.whl
                 pip install https://autogluon.s3.amazonaws.com/dist/autogluon-0.0.1-py3-none-any.whl

           .. container:: cuda

              .. code-block:: bash

                  This option is only available by building from source.

        .. container:: source

           .. container:: cpu

              .. code-block:: bash

                 pip install https://autogluon.s3.amazonaws.com/dist/mxnet-1.6.0b20191125-cp37-cp37m-macosx_10_13_x86_64.whl
                 git clone https://github.com/awslabs/autogluon
                 cd autogluon && python setup.py install --user

           .. container:: cuda

              .. code-block:: bash

                 # Please build mxnet from source manually
                 git clone https://github.com/awslabs/autogluon
                 cd autogluon && python setup.py install --user

