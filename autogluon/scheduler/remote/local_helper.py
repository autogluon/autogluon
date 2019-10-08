import time
import logging
import threading
import subprocess
import multiprocessing as queue

from .ssh_helper import bcolors

class AsyncLineReader(threading.Thread):
    def __init__(self, fd, outputQueue, name, err):
        threading.Thread.__init__(self)
        assert callable(fd.readline)
        self.fd = fd
        self.outputQueue = outputQueue
        self.name = name
        self.err = err

    def run(self):
        while True:
            try:
                line = self.fd.readline()
            except Exception:
                break
            if len(line) > 0:
                line = line.rstrip().decode("utf-8") 
                if self.err:
                    msg = "[\033[1m {name} local\033[0m ] : {output}". \
                        format(name=self.name, output=line)
                else:
                    msg = "[ {name} local] : {output}". \
                        format(name=self.name, output=line)
                if 'distributed.' not in msg:
                    print(msg)
                #self.outputQueue.put(msg)
            else:
                time.sleep(0.1)

    def eof(self):
        return not self.is_alive() and self.outputQueue.empty()

    @classmethod
    def getForFd(cls, fd, name, err=False, start=True):
        stdqueue = queue.Queue()
        reader = cls(fd, stdqueue, name, err)
        if start:
            reader.start()
        return reader, stdqueue

def start_local_worker(master_ip, port):
    process = subprocess.Popen('python -m distributed.cli.dask_worker '
                               '{}:{} --no-nanny'.format(master_ip, port),
                               shell=True,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)
    (stdoutReader, stdoutQueue) = AsyncLineReader.getForFd(process.stdout, name='worker')
    (stderrReader, stderrQueue) = AsyncLineReader.getForFd(process.stderr, name='worker', err=True)
    worker = {'Process': process,
              'stdoutReader': stdoutReader,
              'stdout_queue': stdoutQueue,
              'stderrReader': stderrReader,
              'stderr_queue': stderrQueue}
    return worker


def start_local_scheduler(port):
    process = subprocess.Popen('python -m distributed.cli.dask_scheduler '
                               '--port {}'.format(port),
                               shell=True, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)
    (stdoutReader, stdoutQueue) = AsyncLineReader.getForFd(process.stdout, name='scheduler')
    (stderrReader, stderrQueue) = AsyncLineReader.getForFd(process.stderr, name='scheduler', err=True)
    scheduler = {'Process': process,
                 'stdoutReader': stdoutReader,
                 'stdout_queue': stdoutQueue,
                 'stderrReader': stderrReader,
                 'stderr_queue': stderrQueue}
    return scheduler
