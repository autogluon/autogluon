:::{note}

```{include} install-windows-generic.md
```

```bash
conda create -n myenv python=3.9 cudatoolkit=11.3 -y
conda activate myenv
```

4. Install the proper GPU PyTorch version by following the [PyTorch Install Documentation](https://pytorch.org/get-started/locally/) (Recommended). Alternatively, use the following command:

```bash
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
```

5. Sanity check that your installation is valid and can detect your GPU via testing in Python:

```python
import torch
print(torch.cuda.is_available())  # Should be True
print(torch.cuda.device_count())  # Should be > 0
```

6. Continue with the remaining installation steps using the conda environment created above

:::
