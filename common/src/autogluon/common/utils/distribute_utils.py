import os


class DistributedContext:
    """Class to manage distributed context"""

    @staticmethod
    def get_util_path() -> str:
        """Return the S3 path to store utils generated in distributed training"""
        return os.environ.get("AG_UTIL_PATH")

    @staticmethod
    def get_model_sync_path() -> str:
        """Return the S3 path to sync the model artifacts generated in distributed training"""
        return os.environ.get("AG_MODEL_SYNC_PATH")

    @staticmethod
    def is_distributed_mode() -> bool:
        """Return if the current context is in distributed mode or not"""
        return os.environ.get("AG_DISTRIBUTED_MODE", False)
