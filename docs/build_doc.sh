#!/bin/bash
pip install git+https://github.com/d2l-ai/d2l-book

d2lbook build eval
d2lbook build rst
d2lbook build html
d2lbook build pkg
