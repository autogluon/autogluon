import logging

DEFAULT_LOGGING_LEVEL = 20

logging.basicConfig(format='%(message)s') # just print message in logs
logger = logging.getLogger() # root logger
logger.setLevel(DEFAULT_LOGGING_LEVEL)