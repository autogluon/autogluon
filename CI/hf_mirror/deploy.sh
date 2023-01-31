#!/bin/bash

aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 369469875935.dkr.ecr.us-east-1.amazonaws.com
docker build -t autogluon-hf-mirror .
docker tag autogluon-hf-mirror:latest 369469875935.dkr.ecr.us-east-1.amazonaws.com/autogluon-hf-mirror:latest
docker push 369469875935.dkr.ecr.us-east-1.amazonaws.com/autogluon-hf-mirror:latest