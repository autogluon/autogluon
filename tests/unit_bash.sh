#!/bin/bash

for f in unittests/*.py; do
    python $f
    sleep 1
done
