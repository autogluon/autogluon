.. include:: install-warning.rst

* Which version of MXNet does AutoGluon support?

   Currently, AutoGluon supports MXNet>=1.7.0. In order to ensure that you are installing mxnet
   larger than 1.7.0, you can use

   .. code-block::

     # For CPU
     python3 -m pip install "mxnet<2.0.0, >=1.7.0"

     # For GPU users, CUDA 101
     python3 -m pip install "mxnet_cu101<2.0.0, >=1.7.0"

* I cannot install the package and it reports the error "XXX is not a supported wheel on this platform".

   One possibility is that you are using an older version of pip. Try to upgrade your pip to a version later than "19.0.0", e.g., use the following command:

   .. code-block::

     python3 -m pip install --upgrade pip --user
     python3 -m pip install --upgrade setuptools --user

* On MacOS I am getting a segmentation fault when trying to train LightGBM / XGBoost.

   You need to install libOMP 11 to avoid segmentation faults on MacOS when training LightGBM / XGBoost:

   .. code-block::

      # brew install wget
      wget https://raw.githubusercontent.com/Homebrew/homebrew-core/fb8323f2b170bd4ae97e1bac9bf3e2983af3fdb0/Formula/libomp.rb
      brew uninstall libomp
      brew install libomp.rb
      rm libomp.rb

   For more information, refer to https://github.com/microsoft/LightGBM/issues/4897

* Does AutoGluon support ARM/M1 Mac?
  
  AutoGluon does not officially support ARM/M1 Mac. For more information, refer to https://github.com/autogluon/autogluon/issues/1242

* Why do the install instructions use `pip3` instead of `pip`?

    When you type `pip` in the console, the system looks for an executable file with that name in the current folder and then in the folders specified in the system PATH variable.
    If you have multiple Python installations and all of them are in your PATH, you cannot be sure which directory will be searched first.
    Therefore, if you have Python 2 installed and it is earlier in your PATH, `pip` would refer to `pip2` rather than `pip3`.
    By explicitly using `pip3`, we guard against this issue.
    Further, if Python 4 releases, the install instructions for Python 3 versions of AutoGluon will continue to work even if `pip` begins referring to `pip4`.
    For more information, refer to https://techwithtech.com/python-pip-vs-pip3/

* How to upgrade python version in Kaggle to install AutoGluon 0.7 and later? 

   AutoGluon will drop python 3.7 support in release v0.7 and afterwards. However, the python version is still 3.7 in Kaggle default container, 
   which will lead to installation issue. To upgrade the python version to 3.8 or higher, here is a quick solution following 
   https://www.kaggle.com/code/amareltaylor/how-to-install-alternative-python-version-hack

   .. code-block::
      !conda create -n newPython -c cctbx202208 -y
      !source /opt/conda/bin/activate newPython && conda install -c cctbx202208 python -y
      !/opt/conda/envs/newPython/bin/python3 -m pip install autogluon

   Note that, once AutoGluon is installed, please restart the notebook runtime and then import modules to run your code. 
