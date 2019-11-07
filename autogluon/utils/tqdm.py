from .miscs import in_ipynb

if in_ipynb():
    try:
        from tqdm.notebook import tqdm
    except Exception:
        from tqdm import tqdm
else:
    from tqdm import tqdm

__all__ = ['tqdm']
