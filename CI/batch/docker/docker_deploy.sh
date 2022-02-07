#!/bin/bash

TYPE=$1

if [ -z $TYPE ]; then
	echo "No type detected. Choices: cpu, gpu"
	exit 1
fi;

if [ $TYPE == cpu ] || [ $TYPE == CPU ]; then
	docker build --no-cache -f Dockerfile.cpu -t autogluon-ci:cpu-latest .
	docker tag autogluon-ci:cpu-latest $AWS_ECR_REPO:cpu-latest
	docker push $AWS_ECR_REPO:cpu-latest
elif [ $TYPE == gpu ] || [ $TYPE == GPU ]; then
	docker build --no-cache -f Dockerfile.gpu -t autogluon-ci:gpu-latest .
	docker tag autogluon-ci:gpu-latest $AWS_ECR_REPO:gpu-latest
	docker push $AWS_ECR_REPO:gpu-latest
else
	echo "Invalid type detected. Choices: cpu, gpu"
	exit 1
fi;