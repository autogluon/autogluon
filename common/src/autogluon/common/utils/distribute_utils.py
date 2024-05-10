import os


class DistributedContext:
    """Class to manage distributed context."""

    @staticmethod
    def get_util_path() -> str:
        """Return the path to store utils generated in distributed training. So far only used for HPO.

        Options for the environment variable `AG_UTIL_PATH`:
            - str, an S3 path if using AWS.
        """
        return os.environ.get("AG_UTIL_PATH")

    @staticmethod
    def get_model_sync_path() -> str:
        """Return a path to sync the model artifacts generated in distributed training.

        Options for the environment variable `AG_MODEL_SYNC_PATH`:
            - str, an S3 path if using AWS.
        """
        return os.environ.get("AG_MODEL_SYNC_PATH")


    @staticmethod
    def is_shared_network_file_system() -> bool:
        """Return if the current context is using a shared (network) file system."""
        return os.environ.get("AG_DISTRIBUTED_MODE_NFS", False)

    @staticmethod
    def is_distributed_mode() -> bool:
        """Return if the current context is in distributed mode or not."""
        return os.environ.get("AG_DISTRIBUTED_MODE", False)
