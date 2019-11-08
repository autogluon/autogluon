#!/bin/bash

rm -rf _build
d2lbook build rst
sphinx-autogen -t ./static/ _build/rst/api/*.rst
d2lbook build html
