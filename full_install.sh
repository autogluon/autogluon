deactivate
rm -rf ./venv
python3 -m venv ./venv
source ./venv/bin/activate
pip install -U pip
pip install mxnet-cu102
pip install numpy
pip install gluonnlp==0.8.3
pip install torch
pip install torchvision
pip install wheel

# Mac-only fix
if [[ "$OSTYPE" == "darwin"* ]]; then
  pip uninstall typing -y
fi

pip install -e core/
pip install -e tabular/
pip install -e mxnet/
pip install -e extra/
pip install -e text/
pip install -e vision/
pip install -e autogluon/

echo "!!!! IMPORTANT !!!! Run 'source ./venv/bin/activate' to activate python environment"