from .constants import SAGEMAKER_RESOURCE_PREFIX

TRUST_RELATIONSHIP_FILE_NAME = "ag_cloud_sagemaker_trust_relationship.json"
IAM_POLICY_FILE_NAME = "ag_cloud_sagemaker_iam_policy.json"

TRUST_RELATIONSHIP_ACCOUNT_PLACE_HOLDER = "ACCOUNT"
POLICY_ACCOUNT_PLACE_HOLDER = "ACCOUNT"
POLICY_BUCKET_PLACE_HOLDER = "CLOUD_BUCKET"

SAGEMAKER_TRUST_RELATIONSHIP = {
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "",
            "Effect": "Allow",
            "Principal": {
                "Service": "sagemaker.amazonaws.com",
                "AWS": f"arn:aws:iam::{POLICY_ACCOUNT_PLACE_HOLDER}:root",
            },
            "Action": "sts:AssumeRole",
        }
    ],
}

SAGEMAKER_CLOUD_POLICY_NAME = "AutoGluonSageMakerCloudPredictor"
SAGEMAKER_CLOUD_POLICY_DESCRIPTION = "AutoGluon CloudPredictor with SageMaker Backend Required Policy"

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
                "sagemaker:InvokeEndpoint",
            ],
            "Resource": [
                f"arn:aws:sagemaker:*:{POLICY_ACCOUNT_PLACE_HOLDER}:endpoint/{SAGEMAKER_RESOURCE_PREFIX}*",
                f"arn:aws:sagemaker:*:{POLICY_ACCOUNT_PLACE_HOLDER}:endpoint-config/{SAGEMAKER_RESOURCE_PREFIX}*",
                f"arn:aws:sagemaker:*:{POLICY_ACCOUNT_PLACE_HOLDER}:model/autogluon-inference*",
                f"arn:aws:sagemaker:*:{POLICY_ACCOUNT_PLACE_HOLDER}:training-job/{SAGEMAKER_RESOURCE_PREFIX}*",
                f"arn:aws:sagemaker:*:{POLICY_ACCOUNT_PLACE_HOLDER}:transform-job/{SAGEMAKER_RESOURCE_PREFIX}*",
            ],
        },
        {
            "Sid": "IAM",
            "Effect": "Allow",
            "Action": ["iam:PassRole"],
            "Resource": [
                f"arn:aws:iam::{POLICY_ACCOUNT_PLACE_HOLDER}:role/*",
            ],
        },
        {
            "Sid": "CloudWatchDescribe",
            "Effect": "Allow",
            "Action": ["logs:DescribeLogStreams"],
            "Resource": [f"arn:aws:logs:*:{POLICY_ACCOUNT_PLACE_HOLDER}:log-group:*"],
        },
        {
            "Sid": "CloudWatchGet",
            "Effect": "Allow",
            "Action": ["logs:GetLogEvents"],
            "Resource": [f"arn:aws:logs:*:{POLICY_ACCOUNT_PLACE_HOLDER}:log-group:*:log-stream:*"],
        },
        {
            "Sid": "S3Object",
            "Effect": "Allow",
            "Action": [
                "s3:PutObject",
                "s3:PutObjectAcl",
                "s3:GetObject",
                "s3:GetObjectAcl",
                "s3:AbortMultipartUpload",
            ],
            "Resource": [
                f"arn:aws:s3:::{POLICY_BUCKET_PLACE_HOLDER}/*",
                f"arn:aws:s3:::{POLICY_BUCKET_PLACE_HOLDER}",
                "arn:aws:s3:::*SageMaker*",
                "arn:aws:s3:::*Sagemaker*",
                "arn:aws:s3:::*sagemaker*",
            ],
        },
        {
            "Effect": "Allow",
            "Action": [
                "s3:CreateBucket",
                "s3:GetBucketLocation",
                "s3:ListBucket",
                "s3:GetBucketCors",
                "s3:PutBucketCors",
                "s3:GetBucketAcl",
                "s3:PutObjectAcl",
            ],
            "Resource": [
                f"arn:aws:s3:::{POLICY_BUCKET_PLACE_HOLDER}/*",
                f"arn:aws:s3:::{POLICY_BUCKET_PLACE_HOLDER}",
                "arn:aws:s3:::*SageMaker*",
                "arn:aws:s3:::*Sagemaker*",
                "arn:aws:s3:::*sagemaker*",
            ],
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
                "sagemaker:ListModels",
                "sagemaker:ListDomains",
            ],
            "Resource": ["*"],
        },
        {
            "Effect": "Allow",
            "Action": "sagemaker:*",
            "Resource": ["arn:aws:sagemaker:*:*:flow-definition/*"],
            "Condition": {"StringEqualsIfExists": {"sagemaker:WorkteamType": ["private-crowd", "vendor-crowd"]}},
        },
        {
            "Sid": "Others",
            "Effect": "Allow",
            "Action": [
                "ecr:BatchGetImage",
                "ecr:Describe*",
                "ecr:GetAuthorizationToken",
                "ecr:GetDownloadUrlForLayer",
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
            ],
            "Resource": ["*"],
        },
    ],
}


def replace_trust_relationship_place_holder(trust_relationship_document, account_id):
    """Replace placeholder inside template with given values"""
    statements = trust_relationship_document.get("Statement", [])
    for statement in statements:
        for principal in statement["Principal"].keys():
            statement["Principal"][principal] = statement["Principal"][principal].replace(
                TRUST_RELATIONSHIP_ACCOUNT_PLACE_HOLDER, account_id
            )
    return trust_relationship_document


def replace_iam_policy_place_holder(policy_document, account_id=None, bucket=None):
    """Replace placeholder inside template with given values"""
    statements = policy_document.get("Statement", [])
    for statement in statements:
        resources = statement.get("Resource", None)
        if resources is not None:
            if account_id is not None:
                statement["Resource"] = [
                    resource.replace(POLICY_ACCOUNT_PLACE_HOLDER, account_id) for resource in statement["Resource"]
                ]
            if bucket is not None:
                statement["Resource"] = [
                    resource.replace(POLICY_BUCKET_PLACE_HOLDER, bucket) for resource in statement["Resource"]
                ]
    return policy_document
