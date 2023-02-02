#!/bin/bash

rm -rf _build/

sphinx-build -b html . _build/html/
