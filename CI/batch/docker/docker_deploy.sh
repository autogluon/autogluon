#!/bin/bash

TYPE=$1

# This executes a command that logs into ECR for both our CI repo and the AutoGluon DLC container repo.
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 369469875935.dkr.ecr.us-east-1.amazonaws.com
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 763104351884.dkr.ecr.us-east-1.amazonaws.com

if [ -z $TYPE ]; then
	echo "No type detected. Choices: cpu, gpu"
	exit 1
fi;

if [ $TYPE == cpu ] || [ $TYPE == CPU ]; then
	docker build -f Dockerfile.cpu -t autogluon-ci:cpu-latest .
	docker tag autogluon-ci:cpu-latest $AWS_ECR_REPO:cpu-latest
	docker push $AWS_ECR_REPO:cpu-latest
elif [ $TYPE == gpu ] || [ $TYPE == GPU ]; then
	docker build -f Dockerfile.gpu -t autogluon-ci:gpu-latest .
	docker tag autogluon-ci:gpu-latest $AWS_ECR_REPO:gpu-latest
	docker push $AWS_ECR_REPO:gpu-latest
else
	echo "Invalid type detected. Choices: cpu, gpu"
	exit 1
fi;