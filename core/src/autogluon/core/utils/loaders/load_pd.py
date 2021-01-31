import multiprocessing, logging
import pandas as pd
from os import listdir
from os.path import isfile, join
from pandas import DataFrame

from . import load_pointer
from ..savers import save_pointer
from .. import s3_utils, multiprocessing_utils
from .load_s3 import list_bucket_prefix_suffix_s3

logger = logging.getLogger(__name__)


def load(path, delimiter=None, encoding='utf-8', columns_to_keep=None, dtype=None, error_bad_lines=True, header=0,
         names=None, format=None, nrows=None, skiprows=None, usecols=None, low_memory=False, converters=None, 
         filters=None, sample_count=None, worker_count=None, multiprocessing_method='forkserver') -> DataFrame:
    if isinstance(path, list):
        return load_multipart(
            paths=path, delimiter=delimiter, encoding=encoding, columns_to_keep=columns_to_keep,
            dtype=dtype, error_bad_lines=error_bad_lines, header=header, names=names, format=format,
            nrows=nrows, skiprows=skiprows, usecols=usecols, low_memory=low_memory, converters=converters,
            filters=filters,
            worker_count=worker_count,
            multiprocessing_method=multiprocessing_method
        )
    if format is not None:
        pass
    elif path.endswith(save_pointer.POINTER_SUFFIX):
        format = 'pointer'
    elif path[-1] == '/' and s3_utils.is_s3_url(path):  # and path[:2] == 's3'
        format = 'multipart_s3'
    elif path[-1] == '/' and not s3_utils.is_s3_url(path):  # and path[:2] != 's3'
        format = 'multipart_local'
    elif '.parquet' in path or '.pq' in path or path[-1] == '/':
        format = 'parquet'
    else:
        format = 'csv'
        if delimiter is None:
            if path.endswith('.tsv'):
                delimiter = '\t'
                logger.debug(f'File delimiter for {path} inferred as \'\\t\' (tab). If this is incorrect, please manually load the data as a pandas DataFrame.')
            else:
                delimiter = ','
                logger.debug(f'File delimiter for {path} inferred as \',\' (comma). If this is incorrect, please manually load the data as a pandas DataFrame.')

    if format == 'pointer':
        content_path = load_pointer.get_pointer_content(path)
        return load(path=content_path, delimiter=delimiter, encoding=encoding, columns_to_keep=columns_to_keep, dtype=dtype, 
                    error_bad_lines=error_bad_lines, header=header, names=names, format=None, nrows=nrows, skiprows=skiprows, 
                    usecols=usecols, low_memory=low_memory, converters=converters, filters=filters, sample_count=sample_count, 
                    worker_count=worker_count, multiprocessing_method=multiprocessing_method)
    elif format == 'multipart_s3':
        bucket, prefix = s3_utils.s3_path_to_bucket_prefix(path)
        return load_multipart_s3(bucket=bucket, prefix=prefix, columns_to_keep=columns_to_keep, dtype=dtype, filters=filters, 
                                 sample_count=sample_count, worker_count=worker_count, multiprocessing_method=multiprocessing_method)  # TODO: Add arguments!
    elif format == 'multipart_local':
        paths = [join(path, f) for f in listdir(path) if (isfile(join(path, f))) & (f.startswith('part-'))]
        return load_multipart(
            paths=paths, delimiter=delimiter, encoding=encoding, columns_to_keep=columns_to_keep,
            dtype=dtype, error_bad_lines=error_bad_lines, header=header, names=names, format=None,
            nrows=nrows, skiprows=skiprows, usecols=usecols, low_memory=low_memory, converters=converters,
            filters=filters,
            worker_count=worker_count,
            multiprocessing_method=multiprocessing_method,
        )
    elif format == 'parquet':
        try:
            df = pd.read_parquet(path, columns=columns_to_keep, engine='fastparquet')  # TODO: Deal with extremely strange issue resulting from torch being present in package, will cause read_parquet to either freeze or Segmentation Fault when performing multiprocessing
        except:
            df = pd.read_parquet(path, columns=columns_to_keep, engine='pyarrow')
        column_count_full = len(df.columns)
    elif format == 'csv':
        df = pd.read_csv(path, converters=converters, delimiter=delimiter, encoding=encoding, header=header, names=names, dtype=dtype, 
                         error_bad_lines=error_bad_lines, low_memory=low_memory, nrows=nrows, skiprows=skiprows, usecols=usecols)
        column_count_full = len(list(df.columns.values))
        if columns_to_keep is not None:
            df = df[columns_to_keep]
    else:
        raise Exception('file format ' + format + ' not supported!')

    row_count = df.shape[0]

    column_count_trimmed = len(list(df.columns.values))

    if filters is not None:
        if isinstance(filters, list):
            for filter in filters:
                df = filter(df)
        else:
            df = filters(df)

    logger.log(20, "Loaded data from: " +str(path)+" | Columns = "+str(column_count_trimmed)+" / "+
               str(column_count_full)+ " | Rows = "+str(row_count)+" -> "+ str(len(df)))
    return df


def load_multipart_child(chunk):
    path, delimiter, encoding, columns_to_keep, dtype, error_bad_lines, header, names, format, nrows, skiprows, usecols, low_memory, converters, filters = chunk
    df = load(path=path, delimiter=delimiter, encoding=encoding, columns_to_keep=columns_to_keep,
            dtype=dtype, error_bad_lines=error_bad_lines, header=header, names=names, format=format,
            nrows=nrows, skiprows=skiprows, usecols=usecols, low_memory=low_memory, converters=converters,
            filters=filters)
    return df


def load_multipart(paths, delimiter=',', encoding='utf-8', columns_to_keep=None, dtype=None, error_bad_lines=True, header=0,
                   names=None, format=None, nrows=None, skiprows=None, usecols=None, low_memory=False, converters=None,
                   filters=None, worker_count=None, multiprocessing_method='forkserver'):
    cpu_count = multiprocessing.cpu_count()
    workers = int(round(cpu_count))
    if worker_count is not None:
        if worker_count <= workers:
            workers = worker_count

    logger.log(15, 'Load multipart running pool with '+str(workers)+' workers...')

    full_chunks = [[
        path, delimiter, encoding, columns_to_keep, dtype, error_bad_lines, header, names,
        format, nrows, skiprows, usecols, low_memory, converters, filters
    ] for path in paths]

    df_list = multiprocessing_utils.execute_multiprocessing(workers_count=workers, transformer=load_multipart_child,
                                                            chunks=full_chunks, multiprocessing_method=multiprocessing_method)

    df_combined = pd.concat(df_list, axis=0, ignore_index=True)

    column_count = len(list(df_combined.columns.values))
    row_count = df_combined.shape[0]

    logger.log(20, "Loaded data from multipart file | Columns = "+str(column_count)+" | Rows = "+str(row_count))
    return df_combined


# Loads multiple files and concatenates row-wise (adding columns together)
def load_multi(path_list, delimiter=',', encoding='utf-8', columns_to_keep_list=None, dtype_list=None):
    num_files = len(path_list)

    df_list = []
    for i in range(num_files):
        columns_to_keep = None
        dtype = None
        if dtype_list:
            dtype = dtype_list[i]
        if columns_to_keep_list:
            columns_to_keep = columns_to_keep_list[i]
        df = load(path_list[i], delimiter=delimiter, encoding=encoding, columns_to_keep=columns_to_keep, dtype=dtype)
        df_list.append(df)

    df_multi = pd.concat(df_list, axis=1, sort=False)

    column_count = len(list(df_multi.columns.values))
    row_count = df_multi.shape[0]
    logger.log(20, "Loaded data from "+str(num_files)+" files | Columns = "+str(column_count)+" | Rows = "+str(row_count))
    return df_multi


def load_multipart_s3(bucket, prefix, columns_to_keep=None, dtype=None, sample_count=None, filters=None, worker_count=None, multiprocessing_method='forkserver'):
    if prefix[-1] == '/':
        prefix = prefix[:-1]
    files = list_bucket_prefix_suffix_s3(bucket=bucket, prefix=prefix, suffix='/part-')
    files_cleaned = [file for file in files if prefix + '/part-' in file]
    paths_full = [s3_utils.s3_bucket_prefix_to_path(bucket=bucket, prefix=file, version='s3') for file in files_cleaned]
    if sample_count is not None:
        logger.log(15, 'Load multipart s3 taking sample of '+str(sample_count)+' out of '+str(len(paths_full))+' files to load')
        paths_full = paths_full[:sample_count]

    df = load(path=paths_full, columns_to_keep=columns_to_keep, dtype=dtype, filters=filters, 
              worker_count=worker_count, multiprocessing_method=multiprocessing_method)
    return df
