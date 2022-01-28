# Launch AWS Batch Jobs

Once you've correctly configured the AWS CLI, you may use submit-job.py to deploy your job.

#### Requirements

**boto3** is required. To install it:

```shell
pip install boto3
```

You'll also need to configure it so that the script can authenticate you successfully:

https://boto3.amazonaws.com/v1/documentation/api/latest/guide/quickstart.html#configuration

#### Some arguments

* --job-type: job type to launch(CI-CPU/CI-GPU)
* --source-ref: the branch name
* --remote: repository url
* --command: the command you want to execute
* --wait: let the script hang and display status of the required job

Example:

```shell
python3 submit-job.py \
--job-type CI-CPU \
--source-ref master \
--work-dir examples/tabular \
--remote https://github.com/dmlc/gluon-cv \
--command "python3 example_simple_tabular.py" \
--wait
```

For a full list of arguments and their default values:

```shell
python3 submit-job.py -h
```
