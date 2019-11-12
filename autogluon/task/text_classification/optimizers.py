from gluonnlp import optimizer
from ...core import autogluon_object

__all__ = ['BERTAdam']

@autogluon_object()
class BERTAdam(optimizer.BERTAdam):
    pass
