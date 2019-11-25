from gluonnlp import optimizer
from ...core import obj

__all__ = ['BERTAdam']

@obj()
class BERTAdam(optimizer.BERTAdam):
    pass
