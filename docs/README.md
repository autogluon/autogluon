# How to locally build AutoGluon docs

First make sure the package and all dependencies have been locally installed. Then run:

```
cd autogluon/
python3 -m pip install git+https://github.com/zhanghang1989/d2l-book
python3 -m pip install --force-reinstall ipython==7.16
pip install jupyter_sphinx
pip install docutils\<0.16
cd docs/
bash build_doc.sh
```
