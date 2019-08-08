#!/usr/bin/env bash
mkdir ~/data
mkdir ~/data/sentiment_analysis
cd ~/data/sentiment_analysis/
kaggle competitions download -c sentiment-analysis-on-movie-reviews
unzip train.tsv.zip
unzip test.csv.zip
rm *.zip
echo 'DONE'
