# Updating the Docker Image for AWS Batch
This is for AutoGluon Devs to update the CI docker environment.

**IMPORTANT**:
Please push the changes to our github repository if you updated the docker file and pushed the new docker image to our ECR.
The new docker image will take effect even you didn't push the changes to our github repository.
This helps to make sure everyone sees the changes you made and build on top.

To update the docker:

- Update the Dockerfile
- Log into the AWS account that holds the ECR repo on your dev machine.
- Export the AWS account credentials as environment variables
- CD to the same folder as the Dockerfile and execute the following:

```shell
# First export your ecr repo address as a environment variable
export AWS_ECR_REPO=${your_repo}

# Following script will build, tag, and push the image
# For cpu
./docker_deploy.sh cpu
# For gpu
./docker_deploy.sh gpu

```
