#!/bin/bash
# Reference:
# https://naoto0804.github.io/cross_domain_detection/

if [ -z "$1" ]
  then
    echo "extract data in current directory"
  else
    # check if is valid directory
    if [ ! -d $1 ]; then
        echo $1 "is not a valid directory"
        exit 0
    fi
    echo "navigating to" $1 "..."
    cd $1
fi

curl -O http://www.hal.t.u-tokyo.ac.jp/~inoue/projects/cross_domain_detection/datasets/${name}.zip
unzip ${name}.zip
rm ${name}.zip