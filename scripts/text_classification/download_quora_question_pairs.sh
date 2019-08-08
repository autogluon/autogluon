#!/usr/bin/env bash
mkdir ~/data
mkdir ~/data/quora_question_pairs
cd ~/data/quora_question_pairs/
kaggle competitions download -c quora-question-pairs
unzip train.csv.zip
unzip test.csv.zip
rm *.zip
echo 'DONE'
