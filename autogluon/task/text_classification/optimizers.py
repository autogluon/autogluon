from gluonnlp import optimizer
from ...core import *

__all__ = ['BERTAdam']

@autogluon_object()
class BERTAdam(optimizer.BERTAdam):
    pass
