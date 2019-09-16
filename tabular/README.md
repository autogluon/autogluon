# Tabular

AutoML package that adapts autogluon to tabular datasets. Trains and tunes neural networks + boosted tree models to predict a certain column in a table (can handle both classification and regression).

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

# If you run into issues with LightGBM:  
# https://w.amazon.com/bin/view/Grail/NewHire/Rampup/#HLightGBM

```

## Info for original Grail project (F3GrailExperiments) which this project built on top of 

[Wiki](https://w.amazon.com/bin/view/Grail/)

## Code Repository

[Code Repository](https://code.amazon.com/packages/F3GrailExperiments/trees/mainline)

## Required Packages

[F3GrailDataFrameUtilities](https://code.amazon.com/packages/F3GrailDataFrameUtilities/trees/mainline)

# Create Brazil workspace and use F3GrailExperiments
brazil ws create --name F3GrailExperiments
cd F3GrailExperiments/
brazil ws use F3GrailExperiments
