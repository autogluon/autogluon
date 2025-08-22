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
3. You open an issue to discuss any significant work before implementing it. Major new features, submodules, and model contributions need to fit into our overall design philosophy and our users' interests. We also need to evaluate if the contribution is worth the maintenance overhead.

To send us a pull request, please:

1. [Fork the repository](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo) and [clone the source code to your local machine](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository).
2. Modify the source (see details below); please focus on the specific change you are contributing. If you also reformat all the code, it will be hard for us to focus on your change.
3. Commit to your fork using clear commit messages.
4. [Create a pull request](https://github.com/autogluon/autogluon/pulls) ([GitHub Docs](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request)), answering any default questions in the pull request interface.
5. Pay attention to any automated continuous integration (CI) failures reported in the pull request, and stay involved in the conversation.

GitHub provides additional documentation on [forking a repository](https://help.github.com/articles/fork-a-repo/) and
[creating a pull request](https://help.github.com/articles/creating-a-pull-request/).


## Tips for Modifying the Source Code

- Using a fresh virtualenv, install the package via [these instructions](https://auto.gluon.ai/dev/install.html).
Be sure to select the *Source* option from the installation preferences.

- We recommend developing on Linux as this is the primary OS we develop on and is the primary OS used by our users. We also support Windows and MacOS. Try to avoid introducing changes that will only work on a particular OS. Changes to existing code that improve cross-platform compatibility are most welcome!

- Use Python 3.9, 3.10, 3.11 or 3.12 for development, as these are the only versions where AutoGluon is fully functional.

- Please try to avoid introducing additional dependencies / 3rd party packages (except for model contributions). We are currently working to reduce the number of external dependencies of our package. For now, we recommend [lazy-import](https://github.com/autogluon/autogluon/blob/master/common/src/autogluon/common/utils/try_import.py) of external packages if you are adding functionality that you believe will only be used by a small fraction of users.

- All code should adhere to the [PEP8 style](https://www.python.org/dev/peps/pep-0008/).

- (Optional) After you have edited the code, ensure your changes pass the unit tests via the below commands. Note that in practice we don't do this and instead submit the pull request so that our continuous integration on GitHub automatically runs the tests. This is because our unit tests require multiple hours of compute to complete, and thus it isn't practical to run all the tests on a local machine.
```
# optional, not recommended to run all tests on local machine
pytest common/tests
pytest core/tests
pytest features/tests
pytest tabular/tests
pytest multimodal/tests
pytest timeseries/tests
```

- Style check and import sort the code, so it adheres to our code style by running the below command. Note that our checks for tabular, core, and multimodal modules are temporarily disabled.

```
# the below will check for changes that will occur if performing style checks (safe to run)

# Check formatting and the order of imports
for dir in "timeseries" "common" "features"; do
  ruff format --diff $dir
  ruff check --select I $dir
done
```

```
# the below will actively change files to satisfy style checks
# DO NOT run the below commands before running the above commands, as you risk introducing many unintended changes.

for dir in "timeseries" "common" "features"; do
  ruff format $dir
  ruff check --fix --select I $dir
done
```

- After linting, make sure to commit the linting changes, so it appears in your pull request.

- We encourage you to add your own unit tests, but please ensure they run quickly (unit tests should train models on small data-subsample with the lowest values of training iterations and time-limits that suffice to evaluate the intended functionality). You can run a specific unit test within a specific file like this:
```
python3 -m pytest path_to_file::test_mytest
```
Or remove the ::test_mytest suffix to run all tests in the file:
```
python3 -m pytest path_to_file
```

- If using PyCharm, we highly recommend navigating to `Settings/Preferences` | `Build, Execution, Deployment` | `Python Debugger` and enabling `Drop into debugger on failed tests` for simplified test debugging. 

- To otherwise test your code changes, we recommend running AutoGluon on multiple datasets and verifying the code runs correctly and the resulting accuracy of the trained models is not harmed by your change.  One easy way to test is to simply modify the scripts in [`examples/`](https://github.com/autogluon/autogluon/tree/master/examples), or the [tutorial notebooks](https://github.com/autogluon/autogluon/tree/master/docs/tutorials), which already provide datasets.

- Remember to update all existing examples/tutorials/documentation affected by your code changes.

- We also encourage you to contribute new tutorials using AutoGluon for applications you think other users will be interested in. Please see [`docs/tutorials/`](https://github.com/autogluon/autogluon/tree/master/docs/tutorials). All tutorials should be Jupyter notebooks (.ipynb) files.

- After you open your pull request, our CI system will run to check your code and report found errors. This may take a few hours. Please check back and fix any errors encountered at this stage (you can retrigger a new CI check by pushing updated code to the same PR in a new commit).

## Finding Contributions to Work On
Looking at the existing issues is a great way to find something to contribute on. As our project uses the default GitHub issue labels (enhancement/bug/duplicate/help wanted/invalid/question/wontfix), looking at any ['help wanted'](https://github.com/autogluon/autogluon/labels/help%20wanted) issues is a great place to start.

## Model Contributions (Tabular)

If you are interested in contributing a new model to AutoGluon Tabular, refer to our [custom model tutorial](https://auto.gluon.ai/stable/tutorials/tabular/advanced/tabular-custom-model.html) which provides a solid foundation to base your contribution. 
Please be aware that it is very possible for a model to never be merged and for the PR to be closed for any number of reasons. 
New model contributions have a **very** high bar for acceptance, and will often take months before being merged, if it ever becomes merged. 
The value add for the model has to be substantial, as supporting a new model type is a large ongoing maintenance burden. 
In order to evaluate the value a model provides, our developer team will run extensive benchmarking tests. These are currently manual, time-consuming, and require nuanced interpretation of the results.

We are actively working on ways to automate the evaluation of new model contributions, and hope to have this new logic ready by the end of 2025.

## Code of Conduct
This project has adopted the [Amazon Open Source Code of Conduct](https://aws.github.io/code-of-conduct).
For more information see the [Code of Conduct FAQ](https://aws.github.io/code-of-conduct-faq) or contact
opensource-codeofconduct@amazon.com with any additional questions or comments.


## Security Issue Notifications
If you discover a potential security issue in this project we ask that you notify AWS/Amazon Security via our [vulnerability reporting page](http://aws.amazon.com/security/vulnerability-reporting/). Please do **not** create a public GitHub issue for a security vulnerability.


## Instructions for Project Maintainers

The following instructions are for project maintainers with write access to the AutoGluon repository (those with permissions to merge PRs)

### Platform Tests

To spin up the platform tests, which test AutoGluon on Linux, MacOS, and Windows separately for each supported Python version, do the following:

1. Ensure that the PR branch is originating from the `autogluon/autogluon` repository. It cannot originate from your own personal repository or else the platform tests will not trigger. This is for security reasons.
2. Comment on the PR with `/platform_tests` ([Example](https://github.com/autogluon/autogluon/pull/4714#issuecomment-2522270778)) (requires write permissions in AutoGluon repo). It is recommended to run the platform tests only after you have passed the default CI and only for changes that are likely to cause platform specific issues.

### Benchmarking

To spin up automated benchmarking, do the following:

1. Comment on the PR with `/benchmark module=tabular preset=tabular_best benchmark=tabular_full time_limit=1h` to benchmark the tabular module ([Example](https://github.com/autogluon/autogluon/pull/4714#issuecomment-2524696887)).
2. Comment on the PR with `/benchmark module=multimodal preset=multimodal_best benchmark=automm-image time_limit=g4_12x` to benchmark the multimodal module ([Example](https://github.com/autogluon/autogluon/pull/4714#issuecomment-2524199294))

Automated benchmarking should only be performed for changes that are likely to impact performance, as it is computationally intensive.

## Licensing

This project uses the Apache 2.0 license. See the [LICENSE](https://github.com/autogluon/autogluon/blob/master/LICENSE) file for details. We will ask you to confirm the licensing of your contribution.

We may ask you to sign a [Contributor License Agreement (CLA)](http://en.wikipedia.org/wiki/Contributor_License_Agreement) for larger changes.
