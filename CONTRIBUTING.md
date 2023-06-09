# Contributing Guidelines

Thank you for your interest in contributing to our project. Whether it's a bug report, new feature, correction, or additional
documentation, we greatly value feedback and contributions from our community.

Please read through this document before submitting any issues or pull requests to ensure we have all the necessary
information to effectively respond to your bug report or contribution.


## Reporting Bugs/Feature Requests

We welcome you to use the GitHub issue tracker to report bugs or suggest features.

When filing an issue, please check [existing open](https://github.com/autogluon/autogluon/issues), or [recently closed](https://github.com/autogluon/autogluon/issues?utf8=%E2%9C%93&q=is%3Aissue%20is%3Aclosed%20), issues to make sure somebody else hasn't already
reported the issue. Please try to include as much information as you can. Details like these are incredibly useful:

* A reproducible test case or series of steps
* The version of AutoGluon being used, the version of pytorch
* Any modifications you've made relevant to the bug
* Anything unusual about your environment or deployment

Ideally, you can install AutoGluon and its dependencies in a fresh virtualenv to reproduce the bug.

## Contributing via Pull Requests
Code contributions via pull requests are much appreciated. Before sending us a pull request, please ensure that:

1. You are working against the latest source on the *master* branch.
2. You check existing open, and recently merged, pull requests to make sure someone else hasn't addressed the problem already.
3. You open an issue to discuss any significant work - we would hate for your time to be wasted.

To send us a pull request, please:

1. Fork the repository.
2. Modify the source (see details below); please focus on the specific change you are contributing. If you also reformat all the code, it will be hard for us to focus on your change.
3. Ensure local tests pass.
4. Commit to your fork using clear commit messages.
5. Send us a pull request, answering any default questions in the pull request interface.
6. Pay attention to any automated CI failures reported in the pull request, and stay involved in the conversation.
7. To spin up the platform tests, which test autogluon among macos and windows, comment on the PR with `/platform_tests`(You would need write permission to AutoGluon repo). It is recommended to run the platform tests only after you have passed the default CI.

GitHub provides additional document on [forking a repository](https://help.github.com/articles/fork-a-repo/) and
[creating a pull request](https://help.github.com/articles/creating-a-pull-request/).


## Tips for Modifying the Source Code

- Using a fresh virtualenv, install the package via [these instructions](https://auto.gluon.ai/dev/install.html).
Be sure to select the *Source* option from the installation preferences.

- We recommend developing on Linux as this is the only OS where all features are currently 100% functional. Avoid introducing changes that will only work on a particular OS, as we're currently working to support MacOSX and Windows. Changes to existing code that improve cross-platform compatibility are most welcome!

- Use Python 3.8, 3.9 or 3.10 for development, as these are the only versions where AutoGluon is fully functional.

- Please try to avoid introducing additional dependencies on 3rd party packages. We are currently working to reduce the number of external dependencies of our package. For now, we recommend [lazy-import](https://github.com/autogluon/autogluon/blob/master/common/src/autogluon/common/utils/try_import.py) of external package if you are adding functionality that you believe will only be used by small fraction users.

- All code should adhere to the [PEP8 style](https://www.python.org/dev/peps/pep-0008/).

- After you have edited the code, ensure your changes pass the unit tests via:
```
cd common/
pytest
cd ../core/
pytest
cd ../features/
pytest
cd ../tabular/
pytest
cd ../multimodal/
pytest
cd ../timeseries/
pytest
cd ../eda/
isort src tests && black src tests && tox -e lint,format,typecheck,testenv
```

- We encourage you to add your own unit tests, but please ensure they run quickly (unit tests should train models on small data-subsample with the lowest values of training iterations and time-limits that suffice to evaluate the intended functionality). You can run a specific unit test within a specific file like this:
```
python3 -m pytest path_to_file::test_mytest
```
Or remove the ::test_mytest suffix to run all tests in the file:
```
python3 -m pytest path_to_file
```

- To otherwise test your code changes, we recommend running AutoGluon on multiple datasets and verifying the code runs correctly and the resulting accuracy of the trained models is not harmed by your change.  One easy way to test is to simply modify the scripts in [`examples/`](https://github.com/autogluon/autogluon/tree/master/examples), or the [tutorial notebooks](https://github.com/autogluon/autogluon/tree/master/docs/tutorials), which already provide datasets.

- Remember to update all existing examples/tutorials/documentation affected by your code changes.

- We also encourage you to contribute new tutorials or example scripts using AutoGluon for applications you think other users will be interested in. Please see [`docs/tutorials/`](https://github.com/autogluon/autogluon/tree/master/docs/tutorials) or [`examples/`](https://github.com/autogluon/autogluon/tree/master/examples). All tutorials should be Jupyter notebooks converted into markdown (.md) files by running the command `jupyter nbconvert --ClearOutputPreprocessor.enabled=True --to markdown tutorial.ipynb`. This command also clears out any output cells as our build system will rebuild the .ipynb files from the markdown file and execute the notebooks rendering the output on our website. This is especially important for major new functionality. You can also directly edit .md files in a Jupyter notebook via these steps: https://d2l.ai/chapter_appendix-tools-for-deep-learning/jupyter.html#markdown-files-in-jupyter

- After you open your pull request, our CI system will run for little while to check your code and report found errors. Please check back and fix any errors encountered at this stage (you can retrigger a new CI check by pushing updated code to the same PR in a new commit).



## Finding Contributions to Work On
Looking at the existing issues is a great way to find something to contribute on. As our project uses the default GitHub issue labels (enhancement/bug/duplicate/help wanted/invalid/question/wontfix), looking at any ['help wanted'](https://github.com/autogluon/autogluon/labels/help%20wanted) issues is a great place to start.


## Code of Conduct
This project has adopted the [Amazon Open Source Code of Conduct](https://aws.github.io/code-of-conduct).
For more information see the [Code of Conduct FAQ](https://aws.github.io/code-of-conduct-faq) or contact
opensource-codeofconduct@amazon.com with any additional questions or comments.


## Security Issue Notifications
If you discover a potential security issue in this project we ask that you notify AWS/Amazon Security via our [vulnerability reporting page](http://aws.amazon.com/security/vulnerability-reporting/). Please do **not** create a public github issue.


## Licensing

See the [LICENSE](https://github.com/autogluon/autogluon/blob/master/LICENSE) file for our project's licensing. We will ask you to confirm the licensing of your contribution.

We may ask you to sign a [Contributor License Agreement (CLA)](http://en.wikipedia.org/wiki/Contributor_License_Agreement) for larger changes.
