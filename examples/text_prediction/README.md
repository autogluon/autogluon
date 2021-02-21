# Examples for TextPredictor 

## Top Performance in Product Sentiment Classification

Here, we provide the example that shows how to use AutoGluon to achieve top performance in 
[Product Sentiment Classification Hackathon](https://www.machinehack.com/hackathons/product_sentiment_classification_weekend_hackathon_19/leaderboard) 

```
python3 example_product_sentiment.py
```

## Reach Top-5 Performance in Mercari Price Suggestion

Here, we provide the example that shows how to use AutoGluon to achieve top-5 performance in
 [Mercari Price Suggestion](https://www.kaggle.com/c/mercari-price-suggestion-challenge/data).
To run the example, you will need to configure the Kaggle API which will be documented in 
https://github.com/Kaggle/kaggle-api . 

```
kaggle competitions download -c mercari-price-suggestion-challenge

python3 example_mercari_price_suggestion.py
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
|Text (Single) - Test                   | 0.6747 | 0.9472 | 0.8799 (*)  | 0.9047 (*) | 0.8870   |        |         |        |        |
