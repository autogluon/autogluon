import os

# Environment variables of the removed S3 model sync. Setting them in distributed mode raises an error, since fold
# models would otherwise stay on the node that fitted them.
_REMOVED_S3_SYNC_ENV_VARS = ("AG_MODEL_SYNC_PATH", "AG_UTIL_PATH")


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

    @staticmethod
    def raise_if_s3_sync_requested() -> None:
        """Raise if the environment asks for the removed S3 sync of model artifacts between nodes."""
        requested = [name for name in _REMOVED_S3_SYNC_ENV_VARS if os.environ.get(name)]
        if requested:
            raise ValueError(
                f"{', '.join(requested)} is set, but syncing model artifacts between nodes through S3 is no longer "
                f"supported. Distributed mode requires the predictor path to be on a file system shared by all "
                f"nodes. Unset {', '.join(requested)} to continue."
            )
