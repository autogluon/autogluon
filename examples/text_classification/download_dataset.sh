#!/usr/bin/env bash

echo kaggle dataset: nlp-getting-started
echo data-dir: ./data/nlp-getting-started
mkdir -p ./data
mkdir -p ./data/nlp-getting-started
cd ./data/nlp-getting-started/
kaggle competitions download -c nlp-getting-started
echo nlp-getting-started dataset finish.