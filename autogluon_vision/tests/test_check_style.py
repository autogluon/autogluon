from subprocess import Popen, PIPE
import logging


def test_check_style():
    logging.getLogger().setLevel(logging.INFO)
    logging.info("PEP8 Style check")
    flake8_proc = Popen(['flake8', '--count'], stdout=PIPE)
    flake8_out = flake8_proc.communicate()[0]
    lines = flake8_out.splitlines()
    count = int(lines[-1].decode())
    if count > 0:
        logging.warn("%d PEP8 warnings remaining", count)
    if count > 3438:
        logging.warn("Additional PEP8 warnings were introducing, style check fails")
        return 1
    logging.info("Passed")
    return 0

