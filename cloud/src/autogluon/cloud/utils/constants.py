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
                "iam:PassRole"
            ],
            "Resource": [
                f"arn:aws:iam::{POLICY_ACCOUNT_PLACE_HOLDER}:role/*",
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
                "s3:PutObjectAcl",
                "s3:GetObject",
                "s3:GetObjectAcl",
                "s3:AbortMultipartUpload"
            ],
            "Resource": [
                f"arn:aws:s3:::{POLICY_BUCKET_PLACE_HOLDER}/*",
                f"arn:aws:s3:::{POLICY_BUCKET_PLACE_HOLDER}",
                "arn:aws:s3:::*SageMaker*",
                "arn:aws:s3:::*Sagemaker*",
                "arn:aws:s3:::*sagemaker*"
            ]
        },
        {
            "Effect": "Allow",
            "Action": [
                "s3:GetBucketAcl",
                "s3:PutObjectAcl"
            ],
            "Resource": [
                f"arn:aws:s3:::{POLICY_BUCKET_PLACE_HOLDER}/*",
                f"arn:aws:s3:::{POLICY_BUCKET_PLACE_HOLDER}",
                "arn:aws:s3:::*SageMaker*",
                "arn:aws:s3:::*Sagemaker*",
                "arn:aws:s3:::*sagemaker*"
            ]
        },
        {
            "Sid": "S3Bucket",
            "Effect": "Allow",
            "Action": [
                "s3:CreateBucket",
                "s3:GetBucketLocation",
                "s3:ListBucket",
                "s3:ListAllMyBuckets",
                "s3:GetBucketCors",
                "s3:PutBucketCors"
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
        },
        {
            "Effect": "Allow",
            "Action": [
                "sagemaker:CreatePresignedDomainUrl",
                "sagemaker:DescribeDomain",
                "sagemaker:ListDomains",
                "sagemaker:DescribeUserProfile",
                "sagemaker:ListUserProfiles",
                "sagemaker:*App",
                "sagemaker:ListApps"
            ],
            "Resource": [
                "*"
            ]
        },
        {
            "Effect": "Allow",
            "Action": "sagemaker:*",
            "Resource": [
                "arn:aws:sagemaker:*:*:flow-definition/*"
            ],
            "Condition": {
                "StringEqualsIfExists": {
                    "sagemaker:WorkteamType": [
                        "private-crowd",
                        "vendor-crowd"
                    ]
                }
            }
        },
        {
            "Sid": "Others",
            "Effect": "Allow",
            "Action": [
                "application-autoscaling:DeleteScalingPolicy",
                "application-autoscaling:DeleteScheduledAction",
                "application-autoscaling:DeregisterScalableTarget",
                "application-autoscaling:DescribeScalableTargets",
                "application-autoscaling:DescribeScalingActivities",
                "application-autoscaling:DescribeScalingPolicies",
                "application-autoscaling:DescribeScheduledActions",
                "application-autoscaling:PutScalingPolicy",
                "application-autoscaling:PutScheduledAction",
                "application-autoscaling:RegisterScalableTarget",
                "cloudwatch:DeleteAlarms",
                "cloudwatch:DescribeAlarms",
                "cloudwatch:GetMetricData",
                "cloudwatch:GetMetricStatistics",
                "cloudwatch:ListMetrics",
                "cloudwatch:PutMetricAlarm",
                "cloudwatch:PutMetricData",
                "ec2:CreateNetworkInterface",
                "ec2:CreateNetworkInterfacePermission",
                "ec2:CreateVpcEndpoint",
                "ec2:DeleteNetworkInterface",
                "ec2:DeleteNetworkInterfacePermission",
                "ec2:DescribeDhcpOptions",
                "ec2:DescribeNetworkInterfaces",
                "ec2:DescribeRouteTables",
                "ec2:DescribeSecurityGroups",
                "ec2:DescribeSubnets",
                "ec2:DescribeVpcEndpoints",
                "ec2:DescribeVpcs",
                "ecr:BatchCheckLayerAvailability",
                "ecr:BatchGetImage",
                "ecr:CreateRepository",
                "ecr:Describe*",
                "ecr:GetAuthorizationToken",
                "ecr:GetDownloadUrlForLayer",
                "ecr:StartImageScan",
                "elastic-inference:Connect",
                "elasticfilesystem:DescribeFileSystems",
                "elasticfilesystem:DescribeMountTargets",
                "fsx:DescribeFileSystems",
                "iam:ListRoles",
                "kms:DescribeKey",
                "kms:ListAliases",
                "lambda:ListFunctions",
                "logs:CreateLogDelivery",
                "logs:CreateLogGroup",
                "logs:CreateLogStream",
                "logs:DeleteLogDelivery",
                "logs:Describe*",
                "logs:GetLogDelivery",
                "logs:GetLogEvents",
                "logs:ListLogDeliveries",
                "logs:PutLogEvents",
                "logs:PutResourcePolicy",
                "logs:UpdateLogDelivery",
                "sns:ListTopics",
                "tag:GetResources"
            ],
            "Resource": [
                "*"
            ]
        },
        {
            "Action": "iam:CreateServiceLinkedRole",
            "Effect": "Allow",
            "Resource": [
                "arn:aws:iam::*:role/aws-service-role/sagemaker.application-autoscaling.amazonaws.com/AWSServiceRoleForApplicationAutoScaling_SageMakerEndpoint"
            ],
            "Condition": {
                "StringLike": {
                    "iam:AWSServiceName": "sagemaker.application-autoscaling.amazonaws.com"
                }
            }
        },
        {
            "Effect": "Allow",
            "Action": [
                "sns:Subscribe",
                "sns:CreateTopic",
                "sns:Publish"
            ],
            "Resource": [
                "arn:aws:sns:*:*:*SageMaker*",
                "arn:aws:sns:*:*:*Sagemaker*",
                "arn:aws:sns:*:*:*sagemaker*"
            ]
        }
    ]
}

VALID_ACCEPT = [
    'application/x-parquet',
    'text/csv',
    'application/json'
]
