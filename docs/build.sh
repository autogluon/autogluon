#!/bin/bash

rm -rf _build/
rm -rf jupyter_execute/
sphinx-autogen api/*.rst api/*/*.rst -t _templates/autosummary

sphinx-build -b html . _build/
