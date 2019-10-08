# Adapted from: http://eli.thegreenplace.net/2012/01/04/shared-counter-with-pythons-multiprocessing/
import multiprocessing
import multiprocessing.queues

class SharedCounter(object):
    """ A synchronized shared counter.
    """

    def __init__(self, n = 0):
        self.count = multiprocessing.Value('i', n)

    def increment(self, n = 1):
        """ Increment the counter by n (default = 1) """
        with self.count.get_lock():
            self.count.value += n

    @property
    def value(self):
        """ Return the value of the counter """
        return self.count.value


class Queue(multiprocessing.queues.Queue):
    """ A portable implementation of multiprocessing.Queue.

    Because of multithreading / multiprocessing semantics, Queue.qsize() may
    raise the NotImplementedError exception on Unix platforms like Mac OS X
    where sem_getvalue() is not implemented. This subclass addresses this
    problem by using a synchronized shared counter (initialized to zero) and
    increasing / decreasing its value every time the put() and get() methods
    are called, respectively. This not only prevents NotImplementedError from
    being raised, but also allows us to implement a reliable version of both
    qsize() and empty().
    """
    def __init__(self, *args, **kwargs):
        super(Queue, self).__init__(*args, ctx=multiprocessing.get_context(), **kwargs)
        self.size = SharedCounter(0)

    def put(self, *args, **kwargs):
        self.size.increment(1)
        super(Queue, self).put(*args, **kwargs)

    def get(self, *args, **kwargs):
        self.size.increment(-1)
        return super(Queue, self).get(*args, **kwargs)

    def qsize(self):
        """ Reliable implementation of multiprocessing.Queue.qsize() """
        return self.size.value

    def empty(self):
        """ Reliable implementation of multiprocessing.Queue.empty() """
        return not self.qsize()
