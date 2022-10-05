TRUST_RELATIONSHIP_ACCOUNT_PLACE_HOLDER = 'ACCOUNT'
POLICY_ACCOUNT_PLACE_HOLDER = 'ACCOUNT'
POLICY_BUCKET_PLACE_HOLDER = 'CLOUD_BUCKET'

SAGEMAKER_TRUST_RELATIONSHIP = {
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "",
            "Effect": "Allow",
            "Principal": {
                "Service": "sagemaker.amazonaws.com",
                "AWS": f"arn:aws:iam::{POLICY_ACCOUNT_PLACE_HOLDER}:root"
            },
            "Action": "sts:AssumeRole"
        }
    ]
}

SAGEMAKER_CLOUD_POLICY_NAME = 'AutoGluonSageMakerCloudPredictor'
SAGEMAKER_CLOUD_POLICY_DESCRIPTION = 'AutoGluon CloudPredictor with SageMaker Backend Required Policy'

SAGEMAKER_CLOUD_POLICY = {
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "SageMaker",
            "Effect": "Allow",
            "Action": [
                "sagemaker:DescribeEndpoint",
                "sagemaker:DescribeEndpointConfig",
                "sagemaker:DescribeModel",
                "sagemaker:DescribeTrainingJob",
                "sagemaker:DescribeTransformJob",
                "sagemaker:CreateArtifact",
                "sagemaker:CreateEndpoint",
                "sagemaker:CreateEndpointConfig",
                "sagemaker:CreateModel",
                "sagemaker:CreateTrainingJob",
                "sagemaker:CreateTransformJob",
                "sagemaker:DeleteEndpoint",
                "sagemaker:DeleteEndpointConfig",
                "sagemaker:DeleteModel",
                "sagemaker:UpdateArtifact",
                "sagemaker:UpdateEndpoint",
                "sagemaker:InvokeEndpoint"
            ],
            "Resource": [
                f"arn:aws:sagemaker:*:{POLICY_ACCOUNT_PLACE_HOLDER}:artifact/*",
                f"arn:aws:sagemaker:*:{POLICY_ACCOUNT_PLACE_HOLDER}:transform-job/*",
                f"arn:aws:sagemaker:*:{POLICY_ACCOUNT_PLACE_HOLDER}:endpoint/*",
                f"arn:aws:sagemaker:*:{POLICY_ACCOUNT_PLACE_HOLDER}:training-job/*",
                f"arn:aws:sagemaker:*:{POLICY_ACCOUNT_PLACE_HOLDER}:model/*",
                f"arn:aws:sagemaker:*:{POLICY_ACCOUNT_PLACE_HOLDER}:endpoint-config/*"
            ]
        },
        {
            "Sid": "IAM",
            "Effect": "Allow",
            "Action": [
                "iam:CreatePolicy",
                "iam:DeletePolicy",
                "iam:AttachRolePolicy",
                "iam:DetachRolePolicy",
                "iam:ListAttachedRolePolicies",
                "iam:GetRolePolicy",
                "iam:CreateRole",
                "iam:DeleteRole",
                "iam:GetRole",
                "iam:UpdateRole",
                "iam:PassRole"
            ],
            "Resource": [
                f"arn:aws:iam::{POLICY_ACCOUNT_PLACE_HOLDER}:role/*",
                f"arn:aws:iam::{POLICY_ACCOUNT_PLACE_HOLDER}:policy/*"
            ]
        },
        {
            "Sid": "CloudWatchDescribe",
            "Effect": "Allow",
            "Action": [
                "logs:DescribeLogStreams"
            ],
            "Resource": [
                f"arn:aws:logs:*:{POLICY_ACCOUNT_PLACE_HOLDER}:log-group:*"
            ]
        },
        {
            "Sid": "CloudWatchGet",
            "Effect": "Allow",
            "Action": [
                "logs:GetLogEvents"
            ],
            "Resource": [
                f"arn:aws:logs:*:{POLICY_ACCOUNT_PLACE_HOLDER}:log-group:*:log-stream:*"
            ]
        },
        {
            "Sid": "S3Object",
            "Effect": "Allow",
            "Action": [
                "s3:PutObject",
                "s3:GetObjectAcl",
                "s3:GetObject"
            ],
            "Resource": [
                f"arn:aws:s3:::{POLICY_BUCKET_PLACE_HOLDER}/*",
                "arn:aws:s3:::sagemaker-*/*"
            ]
        },
        {
            "Sid": "S3Bucket",
            "Effect": "Allow",
            "Action": [
                "s3:CreateBucket"
            ],
            "Resource": [
                "arn:aws:s3:::*"
            ]
        },
        {
            "Sid": "ListEvents",
            "Effect": "Allow",
            "Action": [
                "s3:ListAllMyBuckets",
                "sagemaker:ListEndpointConfigs",
                "sagemaker:ListEndpoints",
                "sagemaker:ListTransformJobs",
                "sagemaker:ListTrainingJobs",
                "sagemaker:ListModels"
            ],
            "Resource": [
                "*"
            ]
        }
    ]
}

VALID_ACCEPT = [
    'application/x-parquet',
    'text/csv',
    'application/json'
]
