import json
import logging

logger = logging.getLogger(__name__)


def load(path, *, verbose=True):
    if verbose:
        logger.log(15, 'Loading: %s' % path)
    with open(path, "r") as f:
        out = json.load(f)
    return out
