# How to locally build AutoGluon docs

Instructions apply to Linux and Mac. Windows has not been tested for doc builds.

For MacOS, you need to install pandoc:

```
brew install pandoc
```

Ensure you have a local AutoGluon install for development. If not run the following in package root:

```
pip install -U pip wheel
./full_install.sh
```

Then run in package root:

```
# Note: GPU & CUDA is required to build tutorials
# To skip running tutorials, manually edit `docs/config.ini` and set `eval_notebook = False`
cd docs/
python3 -m pip install -r requirements_doc.txt
python3 -m pip install git+https://github.com/zhanghang1989/d2l-book
```

Now you are ready to run the doc build:

```
bash build_doc.sh
```
