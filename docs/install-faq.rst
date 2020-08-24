* I cannot install the package and it reports the error "XXX is not a supported wheel on this platform".

   One possibility is that you are using an older version of pip. Try to upgrade your pip to a version later than "19.0.0",
e.g., use the following command:

   .. code-block::

     python3 -m pip install --upgrade pip --user

* I see the error "ERROR: No matching distribution found for mxnet<2.0.0,>=1.7.0b20200713".

   It might be due to the out-dated pip version. Try to upgrade the pip via:

   .. code-block::

     python3 -m pip install --upgrade pip --user

* How can I install the customized mxnet (incubating) on SageMaker Notebook?

   You should choose the **conda_python3** kernel and then install the MXNet via

   .. code-block::

     # For CPU users
     python3 -m pip install -U --pre "mxnet>=1.7.0b20200713, <2.0.0" -f https://sxjscience.github.io/KDD2020/

     # For GPU users, CUDA 101
     python3 -m pip install -U --pre "mxnet_cu101>=1.7.0b20200713, <2.0.0" -f https://sxjscience.github.io/KDD2020/

* While running AutoGluon, I get error message "Check failed: e == cudaSuccess: CUDA: initialization error".

  You may have the wrong version of MXNet installed for your CUDA version.
  Match the CUDA version carefully when following the installation instructions.
