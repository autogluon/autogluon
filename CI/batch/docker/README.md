# Updating the Docker Image for AWS Batch
This is for AutoGluon Devs to update the CI docker environment

To update the docker:

- Update the Dockerfile
- Export the AWS account credentials as environment variables
- CD to the same folder as the Dockerfile and execute the following:

```shell
# First export your ecr repo address as a environment variable
export $AWS_ECR_REPO=${your_repo}

# This executes a command that logs into ECR.
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin $AWS_ECR_REPO

# Following script will build, tag, and push the image
# For cpu
./docker_deploy.sh cpu
# For gpu
./docker_deploy.sh gpu

```
