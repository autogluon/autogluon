# Examples for TextPredictor 

## Product Sentiment Classification

Here, we provide the example that shows how to use AutoGluon to achieve top performance in 
[Product Sentiment Classification Hackathon](https://www.machinehack.com/hackathons/product_sentiment_classification_weekend_hackathon_19/leaderboard) 

```
mkdir -p machine_hack_product_sentiment
wget https://automl-mm-bench.s3.amazonaws.com/machine_hack_product_sentiment/all_train.csv -O machine_hack_product_sentiment/all_train.csv
wget https://automl-mm-bench.s3.amazonaws.com/machine_hack_product_sentiment/test.csv -O machine_hack_product_sentiment/test.csv

mkdir -p ag_product_sentiment
python3 run_competition.py --train_file machine_hack_product_sentiment/all_train.csv \
                           --test_file machine_hack_product_sentiment/test.csv \
                           --task product_sentiment \
                           --eval_metric log_loss \
                           --exp_dir ag_product_sentiment \
                           --mode stacking | tee -a ag_product_sentiment/log.txt
```
It will generate a `submission.csv` file and you can try to submit it in 

## Predict Price of Book

```
mkdir -p price_of_books
wget https://automl-mm-bench.s3.amazonaws.com/machine_hack_competitions/predict_the_price_of_books/Data.zip -O price_of_books/Data.zip
cd price_of_books
unzip Data.zip


mkdir -p ag_price_of_books
python3 run_competition.py --train_file price_of_books/Participants_Data/Data_Train.xlsx \
                           --test_file price_of_books/Participants_Data/Data_Test.xlsx \
                           --sample_submission price_of_books/Participants_Data/Sample_Submission.xlsx \
                           --task price_of_books \
                           --eval_metric r2 \
                           --exp_dir ag_price_of_books \
                           --mode stacking | tee -a ag_price_of_books/log.txt
```

## Reach Top-5 Performance in Mercari Price Suggestion

Here, we provide the example that shows how to use AutoGluon to achieve top-5 performance in
 [Mercari Price Suggestion](https://www.kaggle.com/c/mercari-price-suggestion-challenge/data).
To run the example, you will need to configure the Kaggle API which will be documented in 
https://github.com/Kaggle/kaggle-api and download the dataset.

```
sudo apt install -y p7zip-full
bash prepare_mercari_kaggle.sh
```

After you have prepared the dataset, you can use the following command:
```
mkdir -p ag_mercari_price_single
python3 run_competition.py --train_file mercari_price/train.tsv \
                           --test_file mercari_price/test_stg2.tsv \
                           --sample_submission mercari_price/sample_submission_stg2.csv \
                           --task mercari_price \
                           --eval_metric r2 \
                           --exp_dir ag_mercari_price_single \
                           --mode single | tee -a ag_mercari_price_single/log.txt
```

In addition, you may run multimodal with weighted ensemble
```
mkdir -p ag_mercari_price_weighted
python3 run_competition.py --train_file mercari_price/train.tsv \
                           --test_file mercari_price/test_stg2.tsv \
                           --sample_submission mercari_price/sample_submission_stg2.csv \
                           --task mercari_price \
                           --eval_metric r2 \
                           --exp_dir ag_mercari_price_weighted \
                           --mode weighted | tee -a ag_mercari_price_weighted/log.txt
```
Or stacking
```
mkdir -p ag_mercari_price_stacking
python3 run_competition.py --train_file mercari_price/train.tsv \
                           --test_file mercari_price/test_stg2.tsv \
                           --sample_submission mercari_price/sample_submission_stg2.csv \
                           --task mercari_price \
                           --eval_metric r2 \
                           --exp_dir ag_mercari_price_stacking \
                           --mode stacking | tee -a ag_mercari_price_stacking/log.txt
```

## Solve GLUE Tasks with AutoGluon Text

Here, we show how you may use AutoGluon Text to solve all tasks in the [GLUE benchmark](https://openreview.net/pdf?id=rJ4km2R5t7).
 
### Prepare the data
```bash
python3 prepare_glue.py --benchmark glue
```

### Run the benchmark
Run on all datasets with either a single `TextPredictor` model or the `multimodal` configuration 
in AutoGluon Tabular that will use the `TextPredictor` model in the 5-fold-1-layer stacking.
 
```bash
# Run single model
bash run_glue.sh single

# Run 5-fold stacking
bash run_glue.sh stacking
```

### Results
For MRPC and STS, we have manually augmented the training and validation data by shuffling the 
order of two sentences.

|                                       | CoLA   | SST    | MRPC        | STS        | QQP      | MNLI-m | MNLI-mm | QNLI   | RTE    | WNLI   |
|---------------------------------------|--------|--------|-------------|------------|----------|--------|---------|--------|--------|--------|
|Metrics                                | mcc    | acc    | acc         | spearmanr  | f1       | acc    | acc     | acc    | acc    | acc    |
|Text (Single) - Validation (*)         | 0.6747 | 0.9472 | 0.8799 (*)  | 0.9047 (*) | 0.8870   | 0.8643 | 0.8589  | 0.9158 | 0.7726 | 0.5634 |
|Text (Single) - Test                   | -      | -      | -           | -          | -        | -      | -       | -      | -      |        |
