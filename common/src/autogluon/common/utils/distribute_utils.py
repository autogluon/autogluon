import os


class DistributedContext:
    """Class to manage distributed context"""
    
    @staticmethod
    @property
    def model_sync_path() -> str:
        """Return the S3 path to sync the model artifacts generated in distributed training"""
        return os.environ.get("AG_MODEL_SYNC_PATH")
    
    @staticmethod
    @property
    def is_distributed_mode() -> bool:
        """Return if the current context is in distributed mode or not"""
        return os.environ.get("AG_DISTRIBUTED_MODE", False)
