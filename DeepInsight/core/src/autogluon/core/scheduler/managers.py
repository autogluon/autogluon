import multiprocessing as mp

from autogluon.core.scheduler.remote import RemoteManager
from autogluon.core.scheduler.resource import DistributedResourceManager


class TaskManagers(object):

    def __init__(self):
        self._resource_manager = None
        self._remote_manager = None
        self.lock = mp.Lock()

    @property
    def resource_manager(self):
        if self._resource_manager is None:
            self._resource_manager = DistributedResourceManager()
        return self._resource_manager

    @property
    def remote_manager(self):
        if self._remote_manager is None:
            self._remote_manager = RemoteManager()
        return self._remote_manager

    # --------- Delegates ------------

    def register_dist_ip_addrs(self, dist_ip_addrs):
        if dist_ip_addrs is None:
            dist_ip_addrs = []
        self.remote_manager.add_remote_nodes(dist_ip_addrs)
        self.resource_manager.add_remote(self.remote_manager.get_remotes())

    def add_remote(self, ip_addrs):
        """Add remote nodes to the scheduler computation resource.
        """
        ip_addrs = [ip_addrs] if isinstance(ip_addrs, str) else ip_addrs
        with self.lock:
            remotes = self.remote_manager.add_remote_nodes(ip_addrs)
            self.resource_manager.add_remote(remotes)

    def upload_files(self, files, **kwargs):
        """Upload files to remote machines, so that they are accessible by import or load.
        """
        self.remote_manager.upload_files(files, **kwargs)

    def request_resources(self, resources):
        self.resource_manager._request(resources)

    def release_resources(self, resources):
        self.resource_manager._release(resources)

    def check_availability(self, resources):
        return self.resource_manager.check_availability(resources)
