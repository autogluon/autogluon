set -ex

mkdir -p price_of_books
wget https://automl-mm-bench.s3.amazonaws.com/machine_hack_competitions/predict_the_price_of_books/Data.zip -O price_of_books/Data.zip
cd price_of_books
unzip Data.zip
