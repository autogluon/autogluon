import logging
import warnings
from subprocess import Popen, PIPE


def test_check_style():
    logging.getLogger().setLevel(logging.INFO)
    logging.info("PEP8 Style check")
    flake8_proc = Popen(
        ["flake8", "--count", "--exclude", "__init__.py", "--max-line-length", "300"],
        stdout=PIPE,
    )
    flake8_out = flake8_proc.communicate()[0]
    lines = flake8_out.splitlines()
    count = int(lines[-1].decode())
    if count > 0:
        warnings.warn(f"{count} PEP8 warnings remaining\n" + flake8_out.decode())
    assert (
        count < 1000
    ), "Too many PEP8 warnings found, improve code quality to pass test."
