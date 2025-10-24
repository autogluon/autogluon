from __future__ import annotations

import logging
import multiprocessing
import os
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd

from ..utils import multiprocessing_utils, s3_utils

logger = logging.getLogger(__name__)


# TODO: Update so verbose prints at level 20, and adjust calls to save accordingly
# gzip compression produces random deflate issues on linux machines - use with caution
# TODO: v1.0 deprecate 'df', replace with 'data', or align with Pandas for parameter names
# TODO: v1.0 consider renaming function so it isn't 'save'. Consider instead 'save_pd', or something more descriptive.
# TODO: Add full docstring
# TODO: Add `allow_overwrite=True` so that users can instead force
#  an error by setting to False if saving would overwrite an existing file.
def save(
    path: str | Path,
    df: pd.DataFrame,
    index: bool = False,
    verbose: bool = True,
    type: str | None = None,
    sep: str = ",",
    compression: str = "gzip",
    header: bool = True,
):
    """
    Save pandas DataFrame to the file path.
    If local path, directories will be created automatically if necessary to save the file.
        If local, will be relative to working directory unless specified as absolute.
    If S3 path, you must have permissions to save to the S3 location available via boto3 in the current session.

    By default will save the header and index.
    If saving to CSV, column dtypes may not be maintained upon loading the file.
    To ensure identical column dtypes when loading, save in Parquet format.

    For large DataFrames (>1 GB), it is highly recommended to save in Parquet format.
    For massive DataFrames (>10 GB), it is highly recommended to save in multipart Parquet format.

    When saving to multipart Parquet on S3, files under the S3 path that already exist will be deleted
    to avoid corrupting the multipart save.
    Please ensure the S3 path is empty or you are ok with the files being deleted.

    Example paths:
        CSV (local)                 : "local/path/out.csv"
        Parquet (local)             : "local/path/out.parquet"
        Multipart Parquet (local)   : "local/path/"
        CSV (absolute local)        : "/home/ubuntu/path/out.csv"
        Parquet (absolute local)    : "/home/ubuntu/path/out.parquet"
        Multipart Parquet (abs loc) : "/home/ubuntu/path/"
        CSV (S3)                    : "s3://bucket/pre/fix/out.csv"
        Parquet (S3)                : "s3://bucket/pre/fix/out.parquet"
        Multipart Parquet (S3)      : "s3://bucket/pre/fix/"

    Note: Once saved via this function, the same path can be used to
    load the file via `autogluon.common.loaders.load_pd.load(path=path)`.

    """
    if isinstance(path, Path):
        path = str(path)
    if type is None:
        if path[-1] == "/" and s3_utils.is_s3_url(path):  # and path[:2] == 's3'
            type = "multipart_s3"
        elif path[-1] == "/" and not s3_utils.is_s3_url(path):  # and path[:2] != 's3'
            type = "multipart_local"
        elif ".csv" in path:
            type = "csv"
        elif ".parquet" in path:
            type = "parquet"
        else:
            type = "csv"
    if "s3" not in path[:2]:
        is_local = True
    else:
        is_local = False
    column_count = len(list(df.columns.values))
    row_count = df.shape[0]
    if is_local:
        path_abs = os.path.abspath(path)
        path_abs_dirname = os.path.dirname(path_abs)
        if path_abs_dirname:
            os.makedirs(path_abs_dirname, exist_ok=True)
    if type == "csv":
        if is_local:
            df.to_csv(path, index=index, sep=sep, header=header)
        else:
            import boto3

            buffer = StringIO()
            df.to_csv(buffer, index=index, sep=sep, header=header)
            bucket, prefix = s3_utils.s3_path_to_bucket_prefix(s3_path=path)
            s3_resource = boto3.resource("s3")
            s3_resource.Object(bucket, prefix).put(Body=buffer.getvalue(), ACL="bucket-owner-full-control")
        if verbose:
            logger.log(15, "Saved " + str(path) + " | Columns = " + str(column_count) + " | Rows = " + str(row_count))
    elif type == "parquet":
        df.to_parquet(path, compression=compression)
        if verbose:
            logger.log(15, "Saved " + str(path) + " | Columns = " + str(column_count) + " | Rows = " + str(row_count))
    elif type == "multipart_s3":
        bucket, prefix = s3_utils.s3_path_to_bucket_prefix(s3_path=path)
        # We delete the prior files because imagine we are saving a 10-part multipart parquet now,
        #  but we saved a 20 part multipart parquet prior.
        #  In this situation, the output would be corrupted, since the first 10 parts would be the new output,
        #  while parts 11-20 would be the old output.
        #  Multipart Parquet loading would see 20 parts and try to load all of them, resulting in at best an exception,
        #  and at worst the unintended and silent concatenation of two different DataFrames.
        s3_utils.delete_s3_prefix(bucket=bucket, prefix=prefix)  # TODO: Might only delete the first 1000!
        _save_multipart(
            path=path,
            df=df,
            index=index,
            verbose=verbose,
            type="parquet",
            sep=sep,
            compression=compression,
            header=header,
        )
    elif type == "multipart_local":
        # TODO: v1.0 : Ensure the same file deletion process best practice occurs during multipart local saving.
        if os.path.isdir(path):
            for file in os.listdir(path):
                file_path = os.path.join(path, file)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                except Exception as e:
                    logger.exception(e)
        _save_multipart(
            path=path,
            df=df,
            index=index,
            verbose=verbose,
            type="parquet",
            sep=sep,
            compression=compression,
            header=header,
        )
    else:
        raise Exception("Unknown save type: " + type)


def _save_multipart_child(chunk):
    path, df, index, verbose, type, sep, compression, header = chunk
    save(
        path=path,
        df=df,
        index=index,
        verbose=verbose,
        type=type,
        sep=sep,
        compression=compression,
        header=header,
    )


def _save_multipart(path, df, index=False, verbose=True, type=None, sep=",", compression="snappy", header=True):
    cpu_count = multiprocessing.cpu_count()
    workers_count = int(round(cpu_count))
    parts = workers_count

    logger.log(15, "Save_multipart running pool with " + str(workers_count) + " workers")

    paths = [path + "part-" + "0" * (5 - min(5, len(str(i)))) + str(i) + ".parquet" for i in range(parts)]
    df_parts = np.array_split(df, parts)

    full_chunks = [
        [
            path,
            df_part,
            index,
            verbose,
            type,
            sep,
            compression,
            header,
        ]
        for path, df_part in zip(paths, df_parts)
    ]

    multiprocessing_utils.execute_multiprocessing(
        workers_count=workers_count, transformer=_save_multipart_child, chunks=full_chunks
    )

    logger.log(15, "Saved multipart file to " + str(path))
