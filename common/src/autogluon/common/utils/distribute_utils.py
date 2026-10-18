import os


class DistributedContext:
    """Class to manage distributed context based on environment variables.

    In distributed mode, AutoGluon connects to an existing multi-node Ray cluster. Every node reads and writes
    artifacts under the predictor's path, so that path must be on a file system shared by all nodes (for example
    NFS, as used on SLURM clusters).

    Environment variables
    ---------------------
    AG_DISTRIBUTED_MODE: str
        Determines if the current context is in distributed mode or not.
        Must be set to any value to enable distributed mode.
    """

    @staticmethod
    def is_distributed_mode() -> bool:
        """Return if the current context is in distributed mode or not."""
        return os.environ.get("AG_DISTRIBUTED_MODE", False) is not False
