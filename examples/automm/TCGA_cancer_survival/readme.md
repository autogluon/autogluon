# Use AutoGluon TabularPredictor to Predict Vital Status from Patients with Cancer

### 1. Task and dataset

This is a simple example of using AutoGluon on [The Cancer Genome Atlas (TCGA) portal](https://portal.gdc.cancer.gov/). In this example, we leveraged the clinical information to predict whether patients have survived from Head and Neck Squamous Cell Carcinoma or not. More details about this dataset and the task is available at [TCGA-HNSC](https://portal.gdc.cancer.gov/projects/TCGA-HNSC). We shows how to combine the tabular deep learning (DL) model in AutoMM and other tree models via the auto-ensembling logic in AutoGluon-Tabular.

### 2. Run experiments:

```bash
# Default AutoGluon Hyperparameters 
python3 example_kaggle_house.py --automm-mode mlp --mode single 
```

### 3. Results:
