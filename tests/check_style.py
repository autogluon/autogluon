#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Check style"""
import sys
from subprocess import Popen, PIPE
import logging


def main():
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


if __name__ == '__main__':
    sys.exit(main())
