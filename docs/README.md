# How to perform a new release

Refer to `docs/ReleaseInstructions.md`

# How to locally build AutoGluon docs

Instructions apply to Linux and Mac. Windows has not been tested for doc builds.

For MacOS, you need to install pandoc:

```shell
brew install pandoc
```

Ensure you have a local AutoGluon install for development. If not run the following in package root:

```shell
pip install -U pip wheel
./full_install.sh
```

Then run in package root: 

```shell
# Note: GPU & CUDA is required to build tutorials
# To skip running tutorials, manually edit `docs/config.ini` and set `eval_notebook = False`
cd docs/
python3 -m pip install -r requirements_doc.txt
python3 -m pip install git+https://github.com/zhanghang1989/d2l-book
```

Now you are ready to run the doc build:

```shell
bash build_doc.sh
```

## Revise the tutorials

Part of our tutorials are automatically generated and we follow the [D2L standard](https://d2l.ai/chapter_appendix-tools-for-deep-learning/jupyter.html#advanced-options) in developing the notebooks.

To ensure that the tutorial contents are easy for code review, we adopt the Markdown format. You may directly open markdown files when you launch the jupyter notebook via

```shell
python3 -m pip install mu-notedown
jupyter notebook --NotebookApp.contents_manager_class='notedown.NotedownContentsManager'
```

To convert the Jupyter notebook file to markdown file, you can execute the following command:

```shell
jupyter nbconvert --ClearOutputPreprocessor.enabled=True --to markdown YOUR_NOTE_BOOK_FILENAME.ipynb
```
