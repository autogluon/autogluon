# Use AutoMM to Predict California House Prices

Example shows how to combine the tabular deep learning (DL) model in AutoMM and other tree models via the auto-ensembling logic in AutoGluon-Tabular 
to train a good model on the [Kaggle: California House Price Competition](https://www.kaggle.com/c/california-house-prices).

The task is to predict house sale prices based on the house information, such as # of bedrooms, living areas, locations, 
near-by schools, and the seller summary. The data consist of houses sold in California on 2020, with houses in the 
test dataset sold after the ones in the training dataset.

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

For more details about the advanced tabular DL models in AutoMM, you may check [tabular_dl](../tabular_dl).
