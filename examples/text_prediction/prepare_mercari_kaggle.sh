set -ex

kaggle competitions download -c mercari-price-suggestion-challenge
mkdir -p mercari_price
mv mercari-price-suggestion-challenge.zip mercari_price/
cd mercari_price/
unzip mercari-price-suggestion-challenge.zip
7za e train.tsv.7z
unzip test_stg2.tsv.zip
unzip sample_submission_stg2.csv.zip
cd ..
