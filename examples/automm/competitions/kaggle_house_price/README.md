# Use AutoMM to Predict California House Prices
Competition in https://www.kaggle.com/c/california-house-prices

```
kaggle competitions download -c california-house-prices
unzip california-house-prices.zip -d california-house-prices
```

Run experiments:

```
# Single AutoMMPredictor (MLP)
python3 example_kaggle_house.py --automm-mode mlp --mode single 2>&1 | tee -a automm_single_mlp/log.txt

# Single AutoMMPredictor (FT-Transformer For Tabular)
python3 example_kaggle_house.py --automm-mode ft-transformer --mode single 2>&1 | tee -a automm_single_ft/log.txt

# AutoMMPredictor + 5-Fold Bagging
python3 example_kaggle_house.py --automm-mode ft-transformer --mode automm_bag5 2>&1 | tee -a automm_ft_bag5/log.txt

# AutoMMPredictor + other Tree Models (Weighted Ensemble) 
python3 example_kaggle_house.py --automm-mode ft-transformer --mode weighted 2>&1 | tee -a automm_ft_weighted/log.txt

# AutoMMPredictor + other Tree Models (Stack Ensemble) 
python3 example_kaggle_house.py --automm-mode ft-transformer --mode stack5 2>&1 | tee -a automm_ft_stack5/log.txt
```
