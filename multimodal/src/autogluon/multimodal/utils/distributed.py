import logging
import os
import time

from pytorch_lightning import Trainer

from autogluon.common.loaders.load_s3 import list_bucket_prefix_suffix_contains_s3
from autogluon.common.utils.s3_utils import (
    download_s3_folder,
    is_s3_url,
    s3_path_to_bucket_prefix,
    upload_file,
    upload_s3_folder,
)

logger = logging.getLogger(__name__)


def sync_checkpoints(path: str, num_nodes: int, sync_path: str):
    """
    Sync checkpoints saved on worker nodes to the master node

    Parameters
    ----------
    path
        Path to sync the checkpoints into. Typically, this should be the path of the predictor
    num_nodes
        Number of nodes in the cluster including the master node
    sync_path
        The path to fetch checkpoints needed to be synced. This should be a valid s3 path
    """
    assert is_s3_url(sync_path), "Please provide a valid s3 path for synchronization"
    bucket, prefix = s3_path_to_bucket_prefix(sync_path)
    logger.info("Waiting for worker nodes to upload checkpoints")
    finished_prefix = prefix + "finished" if prefix.endswith("/") else prefix + f"/finished"
    while len(list_bucket_prefix_suffix_contains_s3(bucket=bucket, prefix=finished_prefix)) < (num_nodes - 1):
        time.sleep(10)
    for i in range(1, num_nodes):
        logger.info(f"Syncing checkpoints from worker node {i}")
        worker_prefix = prefix + str(i) if prefix.endswith("/") else prefix + f"/{i}"
        download_s3_folder(bucket=bucket, prefix=worker_prefix + "/", local_path=path, error_if_exists=False)


def upload_checkpoints(trainer: Trainer, path: str, sync_path: str):
    """
    Upload checkpoints to sync_path from worker nodes

    Parameters
    ----------
    trainer
        The pytorch lightning trainer object
    path
        The path containing the checkpoints. Typically, this should be the path of the predictor
    sync_path
        The path to upload checkpoints needed to be synced. This should be a valid s3 path
    """
    assert is_s3_url(sync_path), "Please provide a valid s3 path for synchronization"
    bucket, prefix = s3_path_to_bucket_prefix(sync_path)
    node_rank = trainer.node_rank
    finished_prefix = prefix + "finished" if prefix.endswith("/") else prefix + f"/finished"
    worker_prefix = prefix + str(node_rank) if prefix.endswith("/") else prefix + f"/{node_rank}"
    upload_s3_folder(bucket=bucket, prefix=worker_prefix, folder_to_upload=path)
    fname = f"{node_rank}.txt"
    open(fname, "w").close()
    upload_file(bucket=bucket, prefix=finished_prefix, file_name=f"./{fname}")
    try:
        os.remove(fname)
    except FileNotFoundError:
        # Already removed by another process
        pass
