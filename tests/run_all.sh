#!/bin/bash
set -euo pipefail
FILES=./tests/unittests/*.py
for f in $FILES
do
  echo "Evaluating $f file..."
  # take action on each file. $f store current file name
  python $f
done
