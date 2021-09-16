#!/usr/bin/env bash
mkdir ~/data
mkdir ~/data/twittersa
cd ~/data/twittersa/
kaggle competitions download -c twitter-sentiment-analysis2
unzip twitter-sentiment-analysis2.zip
