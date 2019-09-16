# Tabular

AutoML package that adapts autogluon to tabular datasets. Trains and tunes neural networks + boosted tree models to predict a certain column in a table (can handle both classification and regression).

For example usage, see:  **autogluon/task/predict_table_column/examples/example_tabular_predictions.py**


## Setup

To run this code locally (on mac), do the following:

```
# Install libomp to support LightGBM package on mac
brew install libomp

# Create virtual env 
python3 -m venv ~/virtual/TabularAutoGluon
source ~/virtual/TabularAutoGluon/bin/activate

# Install Python packages
pip install -r requirements_local.txt
pip install -r requirements.txt
python setup.py install

# Run smoke test to confirm code is working
python src/tabular/sandbox/smoke/binary/run_smoke_binary.py

