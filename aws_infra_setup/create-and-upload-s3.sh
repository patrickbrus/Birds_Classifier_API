#!/bin/bash
BUCKET_NAME=birds-classifier-model-bucket

AWS_ID=$(aws sts get-caller-identity --query Account --output text | cat)
AWS_REGION=$(aws configure get region)

echo "Creating bucket "
aws s3api create-bucket --acl public-read --bucket $BUCKET_NAME --create-bucket-configuration LocationConstraint=$AWS_REGION 

echo "Enable versioning"
aws s3api put-bucket-versioning --bucket $BUCKET_NAME --versioning-configuration Status=Enabled

echo "Add model to bucket"
aws s3 cp ../model s3://$BUCKET_NAME/ --recursive --output text