#!/usr/bin/env bash
mkdir ~/data
mkdir ~/data/shopeeiet
cd ~/data/shopeeiet/
kaggle competitions download -c shopee-iet-machine-learning-competition
unzip Training%20Images.zip
unzip Test%20Images.zip
mv Test test
rm Training%20Images.zip Test%20Images.zip sample%20submission.csv
