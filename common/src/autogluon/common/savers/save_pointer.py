import logging
import os

POINTER_SUFFIX = ".pointer"

logger = logging.getLogger(__name__)


# TODO: Add S3 support
def save(path, content_path, verbose=True):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    f = open(path, "w")
    f.write(content_path)
    f.close()

    if verbose:
        logger.log(15, "Saved pointer file to " + str(path) + " pointing to " + str(content_path))
