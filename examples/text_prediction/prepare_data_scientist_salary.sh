set -ex

mkdir -p data_scientist_salary
wget https://automl-mm-bench.s3.amazonaws.com/machine_hack_competitions/predict_the_data_scientists_salary_in_india_hackathon/Data.zip -O data_scientist_salary/Data.zip
cd data_scientist_salary
unzip Data.zip

