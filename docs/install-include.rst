Select your preferences and run the install command.

.. note::

  Only Linux installation is supported for now (Mac OSX and Windows versions will be available soon).
  AutoGluon requires `Python <https://www.python.org/downloads/release/python-370/>`_ version 3.6 or 3.7.

.. role:: title
.. role:: opt
   :class: option
.. role:: act
   :class: active option

.. container:: install

  .. container:: opt-group

     :title:`OS:`
     :act:`Linux`

     .. raw:: html

        <div class="mdl-tooltip" data-mdl-for="linux">Linux.</div>

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

           .. container:: gpu

              .. code-block:: bash

                 # Here we assume CUDA 10.0 is installed.  You should change the number 
                 # according to your own CUDA version (e.g. mxnet-cu101 for CUDA 10.1).
                 pip install --pre --upgrade mxnet-cu100
                 git clone https://github.com/awslabs/autogluon
                 cd autogluon && python setup.py install --user
