import os

import pandas as pd


def get_directory_size(path: str) -> int:
    """
    Returns the combined size of all files under the path directory in bytes, excluding symbolic links.
    """
    total_size = 0
    for dir_path, dir_names, file_names in os.walk(path):
        for file_name in file_names:
            file_path = os.path.join(dir_path, file_name)
            if not os.path.islink(file_path):  # skip symbolic links
                file_size = os.path.getsize(file_path)
                total_size += file_size
    return total_size


def get_directory_size_per_file(path: str, *, sort_by: str = "size", include_path_in_name: bool = False) -> pd.Series:
    """
    Returns the size of each file under the path directory in bytes, excluding symbolic links.

    Parameters
    ----------
    path : str
        The path to a directory on local disk.
    sort_by : str, default = "size"
        If None, output files will be ordered based on order of search in os.walk(path).
        If "size", output files will be ordered in descending order of file size.
        If "name", output files will be ordered by name in ascending alphabetical order.
    include_path_in_name : bool, default = False
        If True, includes the full path of the file including the input `path` as part of the index in the output pd.Series.
        If False, removes the `path` prefix of the file path in the index of the output pd.Series.

        For example, for a file located at `foo/bar/model.pkl`, with path='foo/'
            If True, index will be `foo/bar/model.pkl`
            If False, index will be `bar/model.pkl`

    Returns
    -------
    pd.Series with index file path and value file size in bytes.
    """
    file_sizes = dict()
    og_dir_path = None
    for dir_path, dir_names, file_names in os.walk(path):
        if og_dir_path is None:
            og_dir_path = dir_path
        for file_name in file_names:
            file_path = os.path.join(dir_path, file_name)
            if not os.path.islink(file_path):  # skip symbolic links
                file_size = os.path.getsize(file_path)
                if include_path_in_name:
                    file_sizes[file_path] = file_size
                else:
                    # remove path from file_path in dictionary
                    file_sizes[file_path.split(og_dir_path, 1)[-1]] = file_size

    file_size_series = pd.Series(file_sizes, name="size")
    if sort_by is None:
        return file_size_series
    elif sort_by == "size":
        return file_size_series.sort_values(ascending=False)
    elif sort_by == "name":
        return file_size_series.sort_index(ascending=True)
    else:
        raise AssertionError(f'sort_by={sort_by} is unknown. Supported values: [None, "size", "name"]')
