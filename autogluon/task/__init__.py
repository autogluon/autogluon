<<<<<<< HEAD
from .base import BaseTask
from .image_classification import ImageClassification 
#from . import object_detection
#from . import text_classification
=======
from . import image_classification
from . import object_detection
from . import base

__all__ = image_classification.__all__ + object_detection.__all__ + base.__all__
>>>>>>> awslabs/master
