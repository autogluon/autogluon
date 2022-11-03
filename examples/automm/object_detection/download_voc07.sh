#!/bin/bash
# Reference:
# Ellis Brown
# https://github.com/amdegroot/ssd.pytorch/blob/master/data/scripts/VOC2007.sh

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

echo "Downloading VOC2007 trainval ..."
# Download the data.
curl -LO http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
echo "Downloading VOC2007 test data ..."
curl -LO http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
echo "Done downloading."

# Extract data
echo "Extracting trainval ..."
tar -xvf VOCtrainval_06-Nov-2007.tar
echo "Extracting test ..."
tar -xvf VOCtest_06-Nov-2007.tar
echo "removing tars ..."
rm VOCtrainval_06-Nov-2007.tar
rm VOCtest_06-Nov-2007.tar