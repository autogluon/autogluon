# How to perform a new release

Refer to `release_instructions/ReleaseInstructions.md` 

# How to locally build AutoGluon docs

Instructions apply to Linux and Mac. Windows has not been tested for doc builds.

Ensure you have a local AutoGluon install for development. If not run the following in package root:

```shell
pip install -U pip wheel
./full_install.sh
```

Then run in package root:

```shell
cd docs/
python3 -m pip install -r requirements_doc.txt
```

Now you are ready to run the doc build:

```shell
# Note: GPU & CUDA is required to build tutorials
# To skip running tutorials, manually edit `docs/conf.py` and set `nb_execution_mode=off`
bash build_doc.sh
```
