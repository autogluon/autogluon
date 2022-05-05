.. note::

    .. include:: install-windows-generic.rst

    .. code-block:: bash

        conda create -n myenv python=3.9 cudatoolkit=11.3 -y
        conda activate myenv

    4. Install the proper GPU PyTorch version by following the `PyTorch Install Documentation <https://pytorch.org/get-started/locally/>`_ (Recommended). Alternatively, use the following command:

    .. code-block:: bash

        pip3 install "torch>=1.0,<1.11+cu113" -f https://download.pytorch.org/whl/cu113/torch_stable.html

    5. Sanity check that your installation is valid and can detect your GPU via testing in Python:

    .. code-block:: python3

       import torch
       print(torch.cuda.is_available())  # Should be True
       print(torch.cuda.device_count())  # Should be > 0

    6. Continue with the remaining installation steps using the conda environment created above
