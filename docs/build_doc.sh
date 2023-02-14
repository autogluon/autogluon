#!/bin/bash

rm -rf _build
d2lbook build rst
d2lbook build html

cp static/images/* _build/html/_static
