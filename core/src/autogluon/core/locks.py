import multiprocessing as mp


class TaskLock(object):
    TASK_ID = mp.Value('i', 0)
    LOCK = mp.Lock()
