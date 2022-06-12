# Use AutoMM to Predict California House Prices
Competition in https://www.kaggle.com/c/california-house-prices

```
kaggle competitions download -c california-house-prices
unzip california-house-prices.zip -d california-house-prices
```

Run experiments:

```
# Single AutoMMPredictor
python3 example_kaggle_house.py --mode single 2>&1 | tee -a automm_single/log.txt

# AutoMMPredictor + 5-Fold Bagging
python3 example_kaggle_house.py --mode automm_bag5 2>&1 | tee -a automm_bag5/log.txt

# AutoMMPredictor + other Tree Models (Weighted Ensemble) 
python3 example_kaggle_house.py --mode weighted 2>&1 | tee -a automm_weighted/log.txt

# AutoMMPredictor + other Tree Models (Stack Ensemble) 
python3 example_kaggle_house.py --mode stack5 2>&1 | tee -a automm_stack5/log.txt
```
