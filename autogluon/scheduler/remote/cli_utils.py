from distributed.deploy import ssh
from distributed.utils import has_keyword, command_has_keyword

def cli_keywords(d: dict, cls=None, cmd=None):
    """Convert a kwargs dictionary into a list of CLI keywords
    Parameters
    ----------
    d: dict
        The keywords to convert
    cls: callable
        The callable that consumes these terms to check them for validity
    cmd: string or object
        A string with the name of a module, or the module containing a
        click-generated command with a "main" function, or the function itself.
        It may be used to parse a module's custom arguments (i.e., arguments that
        are not part of Worker class), such as nprocs from dask-worker CLI or
        enable_nvlink from dask-cuda-worker CLI.
    Examples
    --------
    >>> cli_keywords({"x": 123, "save_file": "foo.txt"})
    ['--x', '123', '--save-file', 'foo.txt']
    >>> from dask.distributed import Worker
    >>> cli_keywords({"x": 123}, Worker)
    Traceback (most recent call last):
    ...
    ValueError: Class distributed.worker.Worker does not support keyword x
    """
    if cls or cmd:
        for k in d:
            if not has_keyword(cls, k) and not command_has_keyword(cmd, k):
                if cls and cmd:
                    raise ValueError(
                        "Neither class %s or module %s support keyword %s"
                        % (typename(cls), typename(cmd), k)
                    )
                elif cls:
                    raise ValueError(
                        "Class %s does not support keyword %s" % (typename(cls), k)
                    )
                else:
                    raise ValueError(
                        "Module %s does not support keyword %s" % (typename(cmd), k)
                    )

    def convert_value(v):
        out = str(v)
        if " " in out and "'" not in out and '"' not in out:
            out = '"' + out + '"'
        return out

    return sum(
        [["--" + k.replace("_", "-"), convert_value(v)] if v != '--no' else ["--no-" + k.replace("_", "-")] for k, v in d.items()], []
    )

ssh.cli_keywords = cli_keywords
