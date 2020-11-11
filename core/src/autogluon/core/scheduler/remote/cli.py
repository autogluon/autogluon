import multiprocessing as mp

import click

from .remote import DaskRemoteService


@click.command(
    help="""Launch an AutoGluon Remote from terminal, example:
    agremote --address 172.31.14.110 --port 8781
    """
)
@click.option(
    "--address",
    default=None,
    type=str,
    help="Specify the ip address.",
)
@click.option(
    "--port",
    default=8786,
    show_default=True,
    type=int,
    help="Specify the port number.",
)
def main(address, port):
    # Dask requirement - add support for when a program which uses multiprocessing has been frozen to produce a Windows executable.
    mp.freeze_support()
    if ('forkserver' in mp.get_all_start_methods()) & (mp.get_start_method(allow_none=True) != 'forkserver'):
        # The CUDA runtime does not support the fork start method;
        # either the spawn or forkserver start method are required to use CUDA in subprocesses.
        # forkserver is used because spawn is still affected by locking issues
        mp.set_start_method('forkserver', force=True)

    service = DaskRemoteService(address, port)
