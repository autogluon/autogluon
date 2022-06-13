# Use AutoMM to Predict California House Prices
Competition in https://www.kaggle.com/c/california-house-prices

```bash
kaggle competitions download -c california-house-prices
unzip california-house-prices.zip -d california-house-prices
```

Run experiments:

```bash
# Single AutoMMPredictor (MLP)
python3 example_kaggle_house.py --automm-mode mlp --mode single 2>&1 | tee -a logs/automm_single_mlp.txt

# Single AutoMMPredictor (FT-Transformer For Tabular)
python3 example_kaggle_house.py --automm-mode ft-transformer --mode single 2>&1 | tee -a logs/automm_single_ft.txt

# AutoMMPredictor + 5-Fold Bagging
python3 example_kaggle_house.py --automm-mode ft-transformer --mode automm_bag5 2>&1 | tee -a logs/automm_ft_bag5.txt

# AutoMMPredictor + other Tree Models (Weighted Ensemble) 
python3 example_kaggle_house.py --automm-mode ft-transformer --mode weighted 2>&1 | tee -a logs/automm_ft_weighted.txt

# AutoMMPredictor + other Tree Models (5-fold Stack Ensemble) 
python3 example_kaggle_house.py --automm-mode ft-transformer --mode stack5 2>&1 | tee -a logs/automm_ft_stack5.txt
```
