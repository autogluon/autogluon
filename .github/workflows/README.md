This README explains the CI design and how you can update the CI workflow.

## Overall Design
The CI is consisted with two parts: 
1. GitHub Action
    * Push and Pull Request events will trigger GitHub Action workflow defined in `.github/workflows/continuous_integration.yml`.
    * It will then checkout code on **master branch of autogluon** if it's a Pull Request event, or checkout code on **corresponding branch of autogluon** if it's a Push event, to fetch the `submit-job.py` script, which will kick off AWS Batch job to run tests/build docs, under `CI/batch`.
2. AWS Batch.
    * AWS Batch job runs in container defined under `CI/batch/docker`.
    For details on how to build and push the docker image, please refer to `CI/batch/docker/README.md`
    * It will execute scripts defined under `.github/workflow_scripts` for various tasks.
    It will only use scripts updated by Pull Request if the author has write permission to our repo. Otherwise, scripts from **master branch of autogluon** will be used.

## How to update the workflows

### Regular Contributors
**IMPORTANT:** You are only able to update the workflow if you have **write permission** to our repo for security concern. You are still able to add/modify unit tests/ docs and see the changes.

### Maintainers
The general idea is that we only checkout workflow from our master branch unless the workflow is from a branch of `autogluon/autogluon`.

For maintainers with write permission, the easiest way to update our workflow is push to a branch under `autogluon/autogluon`.
The Push event will trigger the workflow to reflect your latest changes.
Once you are satisfied with the changes and the CI passed, start a pull request. CI for both Push and Pull Request event will be triggered in the pull request, and **only the one with Push event reflects your latest workflow changes**

If you try to update the workflow from your fork, **only changes under `.github/workflow_scripts`** will be reflected.
