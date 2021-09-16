import multiprocessing, os, boto3, json, logging
from io import StringIO
import numpy as np

from ...utils import s3_utils, multiprocessing_utils

logger = logging.getLogger(__name__)


# TODO: Update so verbose prints at level 20, and adjust calls to save accordingly
# gzip compression produces random deflate issues on linux machines - use with caution
def save(path, df, index=False, verbose=True, type=None, sep=',', compression='gzip', header=True, json_dump_columns=None):
    if json_dump_columns is not None:
        df = df.copy()
        for column in json_dump_columns:
            if column in df.columns.values:
                df[column] = [json.dumps(x[0]) for x in zip(df[column])]
    if type is None:
        if path[-1] == '/' and s3_utils.is_s3_url(path):  # and path[:2] == 's3'
            type = 'multipart_s3'
        elif path[-1] == '/' and not s3_utils.is_s3_url(path):  # and path[:2] != 's3'
            type = 'multipart_local'
        elif '.csv' in path:
            type = 'csv'
        elif '.parquet' in path:
            type = 'parquet'
        else:
            type = 'csv'
    if 's3' not in path[:2]:
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
    if type == 'csv':
        if is_local:
            df.to_csv(path, index=index, sep=sep, header=header)
        else:
            buffer = StringIO()
            df.to_csv(buffer, index=index, sep=sep, header=header)
            bucket, prefix = s3_utils.s3_path_to_bucket_prefix(s3_path=path)
            s3_resource = boto3.resource('s3')
            s3_resource.Object(bucket, prefix).put(Body=buffer.getvalue(), ACL='bucket-owner-full-control')
        if verbose:
            logger.log(15, "Saved " +str(path)+" | Columns = "+str(column_count)+" | Rows = "+str(row_count))
    elif type == 'parquet':
        try:
            df.to_parquet(path, compression=compression, engine='fastparquet')  # TODO: Might be slower than pyarrow in multiprocessing
        except:
            df.to_parquet(path, compression=compression, engine='pyarrow')
        if verbose:
            logger.log(15, "Saved "+str(path)+" | Columns = "+str(column_count)+" | Rows = "+str(row_count))
    elif type == 'multipart_s3':
        bucket, prefix = s3_utils.s3_path_to_bucket_prefix(s3_path=path)
        s3_utils.delete_s3_prefix(bucket=bucket, prefix=prefix)  # TODO: Might only delete the first 1000!
        save_multipart(path=path, df=df, index=index, verbose=verbose, type='parquet', sep=sep, compression=compression, header=header, json_dump_columns=None)
    elif type == 'multipart_local':
        if os.path.isdir(path):
            for file in os.listdir(path):
                file_path = os.path.join(path, file)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                except Exception as e:
                    logger.exception(e)
        save_multipart(path=path, df=df, index=index, verbose=verbose, type='parquet', sep=sep, compression=compression, header=header, json_dump_columns=None)
    else:
        raise Exception('Unknown save type: ' + type)


def save_multipart_child(chunk):
    path, df, index, verbose, type, sep, compression, header, json_dump_columns = chunk
    save(path=path, df=df, index=index, verbose=verbose, type=type, sep=sep, compression=compression, header=header, json_dump_columns=json_dump_columns)


def save_multipart(path, df, index=False, verbose=True, type=None, sep=',', compression='snappy', header=True, json_dump_columns=None):
    cpu_count = multiprocessing.cpu_count()
    workers_count = int(round(cpu_count))
    parts = workers_count

    logger.log(15, 'Save_multipart running pool with '+str(workers_count)+' workers')

    paths = [path + 'part-' + '0' * (5 - min(5, len(str(i)))) + str(i) + '.parquet' for i in range(parts)]
    df_parts = np.array_split(df, parts)

    full_chunks = [[
        path, df_part, index, verbose, type, sep, compression, header, json_dump_columns,
    ] for path, df_part in zip(paths, df_parts)]

    multiprocessing_utils.execute_multiprocessing(workers_count=workers_count, transformer=save_multipart_child, chunks=full_chunks)

    logger.log(15, "Saved multipart file to "+str(path))
