.. note::

    .. include:: install-windows-generic.rst

    .. code-block:: bash

        conda create -n myenv python=3.9 cudatoolkit=11.3 -y
        conda activate myenv

    4. Install the proper GPU PyTorch version by following the `PyTorch Install Documentation <https://pytorch.org/get-started/locally/>`_ (Recommended). Alternatively, use the following command:

    .. code-block:: bash

        pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 torchtext==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu113

    5. Sanity check that your installation is valid and can detect your GPU via testing in Python:

    .. code-block:: python3

       import torch
       print(torch.cuda.is_available())  # Should be True
       print(torch.cuda.device_count())  # Should be > 0

    6. Continue with the remaining installation steps using the conda environment created above
