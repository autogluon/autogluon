SAGEMAKER_TRUST_REPLATIONSHIP = {
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "",
            "Effect": "Allow",
            "Principal": {
                "Service": "sagemaker.amazonaws.com"
            },
            "Action": "sts:AssumeRole"
        }
    ]
}
# FIXME: Trim down permissions to be minimal
SAGEMAKER_POLICIES = [
    'arn:aws:iam::aws:policy/AmazonS3FullAccess',
    'arn:aws:iam::aws:policy/AmazonSageMakerFullAccess'
]

VALID_ACCEPT = [
    'application/x-parquet',
    'text/csv',
    'application/json'
]
