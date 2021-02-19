# Examples for TextPredictor 

## Top Performance in Product Sentiment Classification

Here, we provide the example that shows how to use AutoGluon to achieve top performance in 
[Product Sentiment Classification Hackathon](https://www.machinehack.com/hackathons/product_sentiment_classification_weekend_hackathon_19/leaderboard) 

```

```

## Reach Top-5 Performance in Mercari Price Suggestion

Here, we provide the example that shows how to use AutoGluon to achieve top-5 performance in [Mercari Price Suggestion](https://www.kaggle.com/c/mercari-price-suggestion-challenge/data).


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

|                          | CoLA   | SST | MRPC | STS       | QQP | MNLI | QNLI | RTE |
|--------------------------|--------|-----|------|-----------|-----|------|------|-----|
|Metrics                   | mcc    | acc | acc  | spearmanr | acc | acc  | acc  | acc |
|Text (Single)             | 0.6747 |     |      |           |     |      |      |     |
|Tabular (Stacking)        |        |     |      |           |     |      |      |     |

