import time, logging

logger = logging.getLogger(__name__)

# decorator to calculate duration taken by any function. Logs times at debug level (logging level 10).
def calculate_time(func):
    # added arguments inside the inner1,
    # if function takes any arguments,
    # can be added like this.
    def inner1(*args, **kwargs):
        # storing time before function execution
        begin = time.time()

        output = func(*args, **kwargs)

        # storing time after function execution
        end = time.time()
        logger.debug("Total time taken in " + str(func.__name__)+": "+str(end - begin))

        return output

    return inner1
