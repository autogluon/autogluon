import logging
from autogluon.tabular_to_image.prediction  import ImagePredictions


try:
    from .version import __version__
except ImportError:
    pass


from .DeepInsight  import image_transformer

logging.basicConfig(format='%(message)s')  # just print message in logs