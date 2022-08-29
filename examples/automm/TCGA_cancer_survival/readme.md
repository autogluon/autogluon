Predict Vital Status from Patients with Cancer

### 1. Task and dataset

This is a simple example of using AutoGluon on [The Cancer Genome Atlas (TCGA) portal](https://portal.gdc.cancer.gov/). In this example, we leveraged the clinical information to predict whether patients have survived from Head and Neck Squamous Cell Carcinoma or not. More details about this dataset and the task is available at [TCGA-HNSC](https://portal.gdc.cancer.gov/projects/TCGA-HNSC). We shows how to use AutoGluon TabularPredictor for this task.

### 2. Run experiments:

```bash
# Default AutoGluon Hyperparameters 
python3 example_kaggle_house.py --path ./datset 
```

### 3. Results:

Model | Test accuracy | Validation accuracy | Train time | Test time  
----  | ----  | ----  | ----  | ---- 
NeuralNetTorch |  0.943218 | 0.925676 | 2.700217 | 0.027071
RandomForestGini |  0.940063 | 0.891892 | 0.603893 | 0.108412
LightGBMLarge |  0.908517 | 0.939189 | 1.351058 |  0.014151 
CatBoost  |  0.905363 | 0.939189 | 4.804489 | 0.025413
XGBoost |  0.905363  | 0.925676 | 0.416847  | 0.027664
WeightedEnsemble_L2 |  0.873817 | 0.945946 | 1.636808  | 0.028049

Full leaderboard information is available at ```./results/leaderboard.csv```
