# Welcome to Contributing to AutoGLuon-AutoMM!
 To guarantee code quality and correctness, we do two kinds of testing: 
  - code style check
  - unit testing

## Code Style Check

### Black
[Black](https://black.readthedocs.io/en/stable/usage_and_configuration/the_basics.html) is a PEP 8 compliant opinionated formatter with its own style. We use it to organize our code style. This is important to guarantee code quality as we have many developers work on the same codebase. The continuous integration (CI) would fail if your code doesn't meet `black`'s style.



Before submitting a pull request, you can run `black` locally to format your code. First, install it:

```
pip install "black>=22.3"
```
Then run it:

```
black source_file_or_directory --line-length 119
```
We use line length 119 instead of the default. You can refer to the CI's [black configurations](https://github.com/awslabs/autogluon/blob/master/pyproject.toml).

Note that if using `black` as a plugin in your IDE, the plugin may not use the configuration file `pyproject.toml` in our project. So, you need to configure the IDE plugin separately by following [here](https://black.readthedocs.io/en/stable/usage_and_configuration/the_basics.html#configuration-via-a-file). 



## Unit Testing
 [Unit testing](https://en.wikipedia.org/wiki/Unit_testing) is necessary to automatically examine that our system's components meet their design and behave as intended.
Everytime we add new features/functions, we need to add new unit tests for them. You can browse the files inside `unittests/` and determine where to add the new test cases properly. To run 
all the unit tests:
```
pytest unittests/
```
You can also run unit tests in one file, e.g., `unittests/automm/test_utils.py`:
```
pytest unittests/automm/test_utils.py
```
Furthermore, testing one function is also easy:
```
pytest unittests/automm/test_utils.py -k test_inferring_pos_label
```
