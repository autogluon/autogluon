# Tabular

AutoML package that adapts autogluon to tabular datasets. Trains and tunes neural networks + boosted tree models to predict a certain column in a table (can handle both classification and regression).

For example usage, see:  **autogluon/task/predict_table_column/examples/example_tabular_predictions.py**

For more advanced usage, see: **autogluon/task/predict_table_column/examples/example_advanced_tabular.py**

## Setup

To run this code locally (on Mac), do the following:

```
# Install libomp to support LightGBM package on mac
brew install libomp

# Create virtual env 
python3 -m venv ~/virtual/TabularAutoGluon
source ~/virtual/TabularAutoGluon/bin/activate
pip install pip --upgrade

# Install MXNet (see instructions here: https://mxnet.apache.org/get_started/?version=v1.5.1&platform=linux&language=python&environ=pip&processor=cpu):

pip install mxnet 

#  Install *tabular* module + dependencies:
cd tabular/ # From inside the parent autogluon/ that contains this whole repository
pip install -r requirements.txt
python setup.py install

# Install *autogluon*:
cd ..
python setup.py develop
