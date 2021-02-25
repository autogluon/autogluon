# Examples for TextPredictor 

## Product Sentiment Classification

Here, we provide the example that shows how to use AutoGluon to achieve top performance in MachineHack Product Sentiment Classification Competition. 
To join the hackathon, you can first go to [MachineHack Website](https://www.machinehack.com/hackathon) and switch to "Late Submission" and 
then go to "Product Sentiment Classification".
This will bring you to the link: [link](https://www.machinehack.com/hackathons/product_sentiment_classification_weekend_hackathon_19/leaderboard)

!IMPORTANT, you can not directly access the link and will have to follow the previous steps.

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
                           --mode stacking  2>&1  | tee -a ag_product_sentiment/log.txt
```
It will generate a `submission.csv` file and you can try to submit that to the competition leaderboard. 

## Predict Price of Book
Here, we provide the example that shows how to use AutoGluon to achieve top performance in MachineHack Book Price Prediction Hackathon.
To join the hackathon, you can first go to [MachineHack Website](https://www.machinehack.com/hackathon) and switch to "Active" and 
go to "Predict The Price Of Books".
This will bring you to the [link](https://www.machinehack.com/hackathons/predict_the_price_of_books/overview)

!IMPORTANT, you can not directly access the link and will have to follow the previous steps.

Also, you will need to install `openpyxl` to read from xlsx file.

```
bash prepare_price_of_books.sh
python3 -m pip install openpyxl
mkdir -p ag_price_of_books
python3 run_competition.py --train_file price_of_books/Participants_Data/Data_Train.xlsx \
                           --test_file price_of_books/Participants_Data/Data_Test.xlsx \
                           --sample_submission price_of_books/Participants_Data/Sample_Submission.xlsx \
                           --task price_of_books \
                           --eval_metric r2 \
                           --exp_dir ag_price_of_books \
                           --mode stacking 2>&1  | tee -a ag_price_of_books/log.txt
```
Once the script is finished, you will see a `submission.xlsx` file generate in the 
`ag_price_of_books` folder and you can try to submit that to the competition leaderboard.

!IMPORTANT. Try to run the experiment on a p3.2x instance :).

## Predict Salary of Data Scientists
Here, we provide the example that shows how to use AutoGluon to achieve top performance in MachineHack Data Scientist Salary Prediction Hackathon. 
To join the hackathon, you can first go to [MachineHack Website](https://www.machinehack.com/hackathon) and switch to "Active" and 
go to "Predict The Data Scientists Salary In India Hackathon".
This will bring you to the link: [link](https://www.machinehack.com/hackathons/predict_the_data_scientists_salary_in_india_hackathon/overview)

!IMPORTANT, you can not directly access the link and will have to follow the previous steps.

Also, you will need to install `openpyxl` to read from xlsx file.

```
bash prepare_data_scientist_salary.sh
python3 -m pip install openpyxl
mkdir -p ag_data_scientist_salary
python3 run_competition.py --train_file data_scientist_salary/Data/Final_Train_Dataset.csv \
                           --test_file data_scientist_salary/Data/Final_Test_Dataset.csv \
                           --sample_submission data_scientist_salary/Data/sample_submission.xlsx \
                           --task data_scientist_salary \
                           --eval_metric acc \
                           --exp_dir ag_data_scientist_salary \
                           --mode stacking 2>&1  | tee -a ag_data_scientist_salary/log.txt
```

Once the script is finished, you will see a `submission.xlsx` file generate in the 
`ag_data_scientist_salary` folder and you can try to submit that to the competition leaderboard.

!IMPORTANT. Try to run the experiment on a p3.2x instance :).

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
                           --mode single 2>&1  | tee -a ag_mercari_price_single/log.txt
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
                           --mode weighted 2>&1  | tee -a ag_mercari_price_weighted/log.txt
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
                           --mode stacking 2>&1  | tee -a ag_mercari_price_stacking/log.txt
```

## Solve GLUE Tasks with AutoGluon Text

Here, we show how you may use AutoGluon Text to solve all tasks in the [GLUE benchmark](https://openreview.net/pdf?id=rJ4km2R5t7).
 
### Prepare the data
```bash
python3 prepare_glue.py --benchmark glue
```

### Run the benchmark
Run on all datasets with either a single `TextPredictor` model or the `multimodal` configuration 
in AutoGluon Tabular that will combine the `TextPredictor` model with tabular models from 
AutoGluon-Tabular via a single layer of stack ensembling with 5-fold bagging.
 
```bash
# Run single model
bash run_glue.sh single

# Run 5-fold stacking
bash run_glue.sh stacking
```

To generate the submission file, use `python3 generate_submission.py --prefix autogluon_text --save_dir submission`

### Results
For MRPC and STS, we have manually augmented the training and validation data by shuffling the 
order of two sentences.

|                                       | CoLA   | SST    | MRPC        | STS        | QQP      | MNLI-m | MNLI-mm | QNLI   | RTE    | WNLI   |
|---------------------------------------|--------|--------|-------------|------------|----------|--------|---------|--------|--------|--------|
|Metrics                                | mcc    | acc    | acc         | spearmanr  | f1       | acc    | acc     | acc    | acc    | acc    |
|Text (Single) - Validation (*)         | 0.6782 | 0.9507 | 0.8725 (*)  | 0.9047 (*) | 0.8866   | 0.8671 | 0.8696  | 0.9235 | 0.7798 | 0.5634 |
|Text (Single) - Test (TBA)             | -      | -      | -           | -          | -        | -      | -       | -      | -      |        |
