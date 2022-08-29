Predict Vital Status from Patients with Cancer

### 1. Task and dataset

This is a simple example of using AutoGluon on [The Cancer Genome Atlas (TCGA) portal](https://portal.gdc.cancer.gov/). In this example, we leveraged the clinical information to predict whether patients have survived from Head and Neck Squamous Cell Carcinoma or not. More details about this dataset and the task is available at [TCGA-HNSC](https://portal.gdc.cancer.gov/projects/TCGA-HNSC). We shows how to use AutoGluon TabularPredictor for this task.

### 2. Run experiments:

```bash
# Benchmark on multiple AG models
python3 example_cancer_survival.py --task TCGA_HNSC --mode all_models
# Just on FT_Transformer 
python3 example_cancer_survival.py --task TCGA_HNSC --mode FT_Transformer
```

### 3. Results:

Model(TCGA-HNSC) | Test accuracy | Validation accuracy | Train time | Test time  
----  | ----  | ----  | ----  | ---- 
NeuralNetTorch |  0.943218 | 0.925676 | 2.700217 | 0.027071
RandomForestGini |  0.940063 | 0.891892 | 0.603893 | 0.108412
LightGBMLarge |  0.908517 | 0.939189 | 1.351058 |  0.014151 
CatBoost  |  0.905363 | 0.939189 | 4.804489 | 0.025413
XGBoost |  0.905363  | 0.925676 | 0.416847  | 0.027664
WeightedEnsemble_L2 |  0.873817 | 0.945946 | 1.636808  | 0.028049
FTTransformer | 0.864353 | 0.891892 | 51.305847 | 0.384669

Full leaderboard details are available at ```./results/leaderboard.csv```. Note that the TCGA-HNSC is a very small tabular dataset with only 1k rows and 29 columns. To test the performance on a larger dataset, we have also included the [adult dataset](https://archive.ics.uci.edu/ml/datasets/adult) with 49k instances. ``` python3 example_cancer_survival.py --task adult --mode all_models ```. The results are as follows:

Model(adults) | Test accuracy | Validation accuracy | Train time | Test time  
----  | ----  | ----  | ----  | ---- 
XGBoost |  0.877162 | 0.8872 | 0.697971 | 0.038446
WeightedEnsemble_L2 |  0.876548 | 0.8908 | 42.201000 | 0.316964
CatBoost |  0.874808 | 0.8828 | 4.263231 |  0.016138 
FTTransformer  |  0.859249 | 0.8696 | 221.684279 | 2.576508 
RandomForestEntr | 0.857918 | 0.8620 | 0.979820 | 0.249948
NeuralNetFastAI | 0.857304 | 0.8620  | 32.106148 | 0.137593
NeuralNetTorch |  0.856382  | 0.8588 | 40.264039   | 0.177079 

While Decision Tree-based models are still the top-performing approaches, FT_Transformer beats other deep-learning approaches.
