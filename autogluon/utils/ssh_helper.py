"""ssh helper for starting remote dask scheduler"""
import socket
from distributed.deploy.ssh import async_ssh

__all__ = ['get_ip', 'RemoteDaskNode']

def get_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # doesn't even have to be reachable
        s.connect(('10.255.255.255', 1))
        IP = s.getsockname()[0]
    except:
        IP = '127.0.0.1'
    finally:
        s.close()
    return IP

class RemoteDaskNode(object):
    def __init__(
        self,
        worker_addr,
        #scheduler_addr=None,
        scheduler_port=8786,
        nthreads=0,
        nprocs=1,
        ssh_username=None,
        ssh_port=22,
        ssh_private_key=None,
        nohost=False,
        logdir=None,
        remote_python=None,
        memory_limit=None,
        worker_port=None,
        nanny_port=None,
        remote_dask_worker="distributed.cli.dask_worker",
    ):

        self.scheduler_addr = worker_addr
        self.scheduler_port = scheduler_port
        self.nthreads = nthreads
        self.nprocs = nprocs

        self.ssh_username = ssh_username
        self.ssh_port = ssh_port
        self.ssh_private_key = ssh_private_key

        self.nohost = nohost

        self.remote_python = remote_python

        self.memory_limit = memory_limit
        self.worker_port = worker_port
        self.nanny_port = nanny_port
        self.remote_dask_worker = remote_dask_worker

        # Generate a universal timestamp to use for log files
        import datetime

        if logdir is not None:
            logdir = os.path.join(
                logdir,
                "dask-ssh_" + datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S"),
            )
            print(
                bcolors.WARNING + "Output will be redirected to logfiles "
                'stored locally on individual worker nodes under "{logdir}".'.format(
                    logdir=logdir
                )
                + bcolors.ENDC
            )
        self.logdir = logdir

        # Keep track of all running threads
        self.threads = []

        # Start the scheduler node
        self.scheduler = start_scheduler(
            logdir,
            worker_addr,
            scheduler_port,
            ssh_username,
            ssh_port,
            ssh_private_key,
            remote_python,
        )

        # Start worker nodes
        self.workers = []
        for i, addr in enumerate(worker_addr):
            self.add_worker(addr)

    def monitor_remote_processes(self):

        # Form a list containing all processes, since we treat them equally from here on out.
        all_processes = [self.scheduler] + self.workers

        try:
            while True:
                for process in all_processes:
                    while not process["output_queue"].empty():
                        print(process["output_queue"].get())

                # Kill some time and free up CPU before starting the next sweep
                # through the processes.
                time.sleep(0.1)
            # end while true

        except KeyboardInterrupt:
            pass  # Return execution to the calling process

    def add_worker(self, address):
        self.workers.append(
            start_worker(
                self.logdir,
                self.scheduler_addr,
                self.scheduler_port,
                address,
                self.nthreads,
                self.nprocs,
                self.ssh_username,
                self.ssh_port,
                self.ssh_private_key,
                self.nohost,
                self.memory_limit,
                self.worker_port,
                self.nanny_port,
                self.remote_python,
                self.remote_dask_worker,
            )
        )

    def shutdown(self):
        all_processes = [self.scheduler] + self.workers

        for process in all_processes:
            process["input_queue"].put("shutdown")
            process["thread"].join()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.shutdown()


def start_scheduler(logdir, addr, port, ssh_username, ssh_port,
                    ssh_private_key, remote_python=None):
    cmd = "{python} -m distributed.cli.dask_scheduler --port {port}".format(
        python=remote_python or sys.executable, port=port, logdir=logdir
    )

    # Optionally re-direct stdout and stderr to a logfile
    if logdir is not None:
        cmd = "mkdir -p {logdir} && ".format(logdir=logdir) + cmd
        cmd += "&> {logdir}/dask_scheduler_{addr}:{port}.log".format(
            addr=addr, port=port, logdir=logdir
        )

    # Format output labels we can prepend to each line of output, and create
    # a 'status' key to keep track of jobs that terminate prematurely.
    label = (
        bcolors.BOLD
        + "scheduler {addr}:{port}".format(addr=addr, port=port)
        + bcolors.ENDC
    )

    # Create a command dictionary, which contains everything we need to run and
    # interact with this command.
    input_queue = Queue()
    output_queue = Queue()
    cmd_dict = {
        "cmd": cmd,
        "label": label,
        "address": addr,
        "port": port,
        "input_queue": input_queue,
        "output_queue": output_queue,
        "ssh_username": ssh_username,
        "ssh_port": ssh_port,
        "ssh_private_key": ssh_private_key,
    }

    # Start the thread
    thread = Thread(target=async_ssh, args=[cmd_dict])
    thread.daemon = True
    thread.start()

    return merge(cmd_dict, {"thread": thread})

def start_worker(logdir, scheduler_addr, scheduler_port, worker_addr, nthreads,
    nprocs, ssh_username, ssh_port, ssh_private_key, nohost, memory_limit,
    worker_port, nanny_port, remote_python=None,
    remote_dask_worker="distributed.cli.dask_worker",):

    cmd = (
        "{python} -m {remote_dask_worker} "
        "{scheduler_addr}:{scheduler_port} "
        "--nthreads {nthreads}" + (" --nprocs {nprocs}" if nprocs != 1 else "")
    )

    if not nohost:
        cmd += " --host {worker_addr}"

    if memory_limit:
        cmd += " --memory-limit {memory_limit}"

    if worker_port:
        cmd += " --worker-port {worker_port}"

    if nanny_port:
        cmd += " --nanny-port {nanny_port}"

    cmd = cmd.format(
        python=remote_python or sys.executable,
        remote_dask_worker=remote_dask_worker,
        scheduler_addr=scheduler_addr,
        scheduler_port=scheduler_port,
        worker_addr=worker_addr,
        nthreads=nthreads,
        nprocs=nprocs,
        memory_limit=memory_limit,
        worker_port=worker_port,
        nanny_port=nanny_port,
    )

    # Optionally redirect stdout and stderr to a logfile
    if logdir is not None:
        cmd = "mkdir -p {logdir} && ".format(logdir=logdir) + cmd
        cmd += "&> {logdir}/dask_scheduler_{addr}.log".format(
            addr=worker_addr, logdir=logdir
        )

    label = "worker {addr}".format(addr=worker_addr)

    # Create a command dictionary, which contains everything we need to run and
    # interact with this command.
    input_queue = Queue()
    output_queue = Queue()
    cmd_dict = {
        "cmd": cmd,
        "label": label,
        "address": worker_addr,
        "input_queue": input_queue,
        "output_queue": output_queue,
        "ssh_username": ssh_username,
        "ssh_port": ssh_port,
        "ssh_private_key": ssh_private_key,
    }

    # Start the thread
    thread = Thread(target=async_ssh, args=[cmd_dict])
    thread.daemon = True
    thread.start()

    return merge(cmd_dict, {"thread": thread})
