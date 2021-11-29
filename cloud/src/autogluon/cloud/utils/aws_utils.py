import boto3
import json

from botocore.exceptions import ClientError


def create_sagemaker_role_and_attach_policies(role_name, trust_relationship, policies):
    iam = boto3.client('iam')
    try:
        iam.create_role(
            RoleName=role_name,
            AssumeRolePolicyDocument=json.dumps(trust_relationship),
            Description='AutoGluon CloudPredictor role',

        )
    except ClientError as e:
        if not e.response['Error']['Code'] == 'EntityAlreadyExists':
            raise e
    for policy in policies:
        try:
            iam.attach_role_policy(
                PolicyArn=policy,
                RoleName=role_name
            )
        except ClientError as e:
            iam.delete_role(
                RoleName=role_name
            )
            raise e
    role_arn = iam.get_role(RoleName=role_name)['Role']['Arn']
    return role_arn
