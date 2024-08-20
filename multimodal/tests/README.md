# Welcome to Contributing to AutoGluon Multimodal!

To guarantee code quality and correctness, we do two kinds of testing:

- code style check
- unit testing

## Code Style Check

### ruff

[ruff](https://github.com/astral-sh/ruff) is a PEP 8 compliant opinionated formatter with its own style. We use it to organize our code style. This is important to guarantee code quality as we have many developers work on the same codebase. The continuous integration (CI) would fail if your code doesn't meet `ruff`'s style.

Before submitting a pull request, you can run `ruff` locally to format your code. First, install it:

```
pip install ruff
```

Then run it:

```
ruff format source_file_or_directory --line-length 119
```

We use line length 119 instead of the default. You can refer to the CI's [ruff configurations](https://github.com/autogluon/autogluon/blob/master/pyproject.toml).

Note that if using `ruff` as a plugin in your IDE, the plugin may not use the configuration file `pyproject.toml` in our project. So, you need to configure the IDE plugin separately by following [here](https://docs.astral.sh/ruff/configuration/).


## Unit Testing

[Unit testing](https://en.wikipedia.org/wiki/Unit_testing) is necessary to automatically examine that our system's components meet their design and behave as intended.
Every time we add new features/functions, we need to add new unit tests for them. You can browse the files inside `unittests/` and determine where to add the new test cases properly. To run
all the unit tests:

```
pytest unittests/
```

You can also run unit tests in one file, e.g., `unittests/test_utils.py`:

```
pytest unittests/test_utils.py
```

Furthermore, testing one function is also easy:

```
pytest unittests/test_utils.py -k test_inferring_pos_label
```
