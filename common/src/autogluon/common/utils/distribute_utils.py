import os


class DistributedContext:
    """Class to manage distributed context based on environment variables.

    Note: Paths can be either local or S3 paths.

    Environment variables
    ---------------------
    AG_DISTRIBUTED_MODE: str
        Determines if the current context is in distributed mode or not.
        Must be set to any value to enable distributed mode.
    AG_UTIL_PATH: str
        Path to store utils generated in distributed training. Only used for HPO.
        Not used for local or network file system.
    AG_MODEL_SYNC_PATH: str
        Path to sync the model artifacts generated in distributed training.
        Not used for local or network file system.
    AG_DISTRIBUTED_FILESYSTEM: str
        Determines the file system to use for distributed training.
        By default, a cloud environment is assumed.
        Alternative values are:
            - "NFS": for a network file system, as used on SLURM clusters.
    """

    @staticmethod
    def get_util_path() -> str:
        """Return the S3 path to store utils generated in distributed training. Only used for HPO."""
        return os.environ.get("AG_UTIL_PATH")

    @staticmethod
    def get_model_sync_path() -> str:
        """Return the S3 path to sync the model artifacts generated in distributed training."""
        return os.environ.get("AG_MODEL_SYNC_PATH")

    @staticmethod
    def is_distributed_mode() -> bool:
        """Return if the current context is in distributed mode or not."""
        return os.environ.get("AG_DISTRIBUTED_MODE", False) is not False

    @staticmethod
    def is_shared_network_file_system() -> bool:
        """Return if the current context is using a shared (network) file system."""
        return os.environ.get("AG_DISTRIBUTED_FILESYSTEM", "False") == "NFS"
