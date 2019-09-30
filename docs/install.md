# Installation

To start with AutoGluon hackathon, we could setup the running environment on our own machines. To do so, we'll need you to set up with a Python environment, Jupyter's interactive notebooks, the relevant libraries, and the code needed for the hackathon.

## Setup the Running Environment on Your Own Machine (Under construction)
:label:`sec_ownenv`

### Obtaining Source Codes

The source code package containing all notebooks for the hackathon is available at http://autogluon-hackathon.s3-website-us-west-2.amazonaws.com/autogluon-hackathon.zip. Please download it and extract it into a folder. For example, on Linux/macos, if you have both `wget` and `unzip` installed, you can do it through:  

```bash
wget http://autogluon-hackathon.s3-website-us-west-2.amazonaws.com/autogluon-hackathon.zip
unzip autogluon-hackathon.zip
```

### Installing Running Environment

If you have both `Python 3.6 or newer and pip installed`, the easiest way to install the running environment through pip. Two packages are needed, `autogluon` for all dependencies such as Jupyter and saved code blocks, and `mxnet` for deep learning framework we are using. First install `autogluon` by

```bash
pip install http://autogluon-hackathon.s3-website-us-west-2.amazonaws.com/dist/autogluon-0.0.1+a1806d9-py3-none-any.whl
```

If unfortunately something went wrong, please check

1. Whether you are using the newest pip version. If not, you can upgrade it through `pip install --upgrade pip`
2. Whether you have permission to install the package on your machine. If not, you can install to your home directory by adding a `--user` flag. Such as `pip install http://autogluon-hackathon.s3-website-us-west-2.amazonaws.com/dist/autogluon-0.0.1+a1806d9-py3-none-any.whl --user`

Before installing `mxnet`, please first check if you are able to access GPUs. If so, please go to :ref:`sec_gpu` for instructions to install a GPU-supported `mxnet`. Otherwise, we can install the CPU version, which would be not optimal for this hackathon.  

```bash
pip install mxnet
```

Once both packages are installed, we now open the Jupyter notebook by

```bash
jupyter notebook
```

At this point open http://localhost:8888 (which usually opens automatically) in the browser, then you can view and run the code in each notebook.

### Upgrade to a New Version

You may want to check a new version for both hackathon contents and MXNet from time to time. 

1. We will maintain the up-to-date contents at http://autogluon-hackathon.s3-website-us-west-2.amazonaws.com/autogluon-hackathon.zip. 
2. MXNet can be upgraded by `pip install MXNet -U` as well. 

### GPU Support

:label:`sec_gpu`

By default MXNet is installed without GPU support to ensure that it will run on any computer (including most laptops). However, for this hackathon, we strongly recommend using GPU version of the MXNet. If your computer has NVIDIA graphics cards and has installed [CUDA](https://developer.nvidia.com/cuda-downloads), you should install a GPU-enabled MXNet. 

If you have installed the CPU-only version, then remove it first by

```bash
pip uninstall mxnet
```

Then you need to find the CUDA version you installed. You may check it through `nvcc --version` or `cat /usr/local/cuda/version.txt`. Assume you have installed CUDA 10.0, then you can install the according MXNet version by 

```bash
pip install mxnet-cu100
```

You may change the last digits according to your CUDA version, e.g. `cu101` for CUDA 10.1 and `cu90` for CUDA 9.0. You can find all available MXNet versions by `pip search mxnet`. 

### More Information

- Checkout [beta.mxnet.io](http://beta.mxnet.io/install/index.html) for more options such as ARM devices and docker images.
- [Verify your MXNet installation](https://beta.mxnet.io/install/validate-mxnet.html).
- [Configure MXNet environment variables](https://mxnet.incubator.apache.org/versions/master/faq/env_var.html).