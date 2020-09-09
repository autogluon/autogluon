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
    service = DaskRemoteService(address, port)
