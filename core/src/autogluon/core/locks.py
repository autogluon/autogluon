import multiprocessing as mp


class TaskLock(object):
    TASK_ID = mp.Value('i', 0)
    LOCK = mp.Lock()


class DecoratorLock(object):
    SEED = mp.Value('i', 0)
    LOCK = mp.Lock()

class RemoteLock(object):
    LOCK = mp.Lock()
    REMOTE_ID = mp.Value('i', 0)
