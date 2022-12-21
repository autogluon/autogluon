#!/bin/bash

aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 369469875935.dkr.ecr.us-east-1.amazonaws.com
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 763104351884.dkr.ecr.us-east-1.amazonaws.com
