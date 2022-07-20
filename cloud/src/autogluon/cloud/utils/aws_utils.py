import boto3
import json

from botocore.exceptions import ClientError
from dataclasses import dataclass
from typing import List, Union
from ..version import __version__


@dataclass
class CustomIamPolicy:
    name: str
    document: str
    description: str


def create_iam_role(role_name: str, trust_relationship: str, description: str, **kwargs):
    iam = boto3.client('iam')
    role_arn = None
    try:
        response = iam.create_role(
            RoleName=role_name,
            AssumeRolePolicyDocument=trust_relationship,
            Description=description,
            **kwargs,
        )
        role_arn = response['Role']['Arn']
    except ClientError as e:
        raise e
    return role_arn


def create_iam_policy(policy_name: str, policy_document: str, policy_description, **kwargs):
    iam = boto3.client('iam')
    policy_arn = None
    try:
        response = iam.create_policy(
            PolicyName=policy_name,
            PolicyDocument=policy_document,
            Description=policy_description,
            **kwargs
        )
        policy_arn = response['Policy']['Arn']
    except ClientError as e:
        raise e
    return policy_arn


def setup_sagemaker_role_and_policy(role_name: str, trust_relationship: dict, policies: List[Union[str, CustomIamPolicy]]):
    iam = boto3.client('iam')
    role_arn = None
    # create policies based on cloud module version if passed in a CustomIamPolicy
    policies_to_attach = list()  # this list holds all policies in ARN
    for policy in policies:
        if type(policy) == CustomIamPolicy:
            try:
                policy = create_iam_policy(policy.name, policy.document, policy.description)
            except ClientError as e:
                if not e.response['Error']['Code'] == 'EntityAlreadyExists':
                    raise e
        policies_to_attach.append(policy)
    # create IAM role
    try:
        create_iam_role(
            role_name=role_name,
            trust_relationship=json.dumps(trust_relationship),
            description='AutoGluon CloudPredictor role'
        )
    except ClientError as e:
        if not e.response['Error']['Code'] == 'EntityAlreadyExists':
            raise e
        else:
            # if exists, add missing policies if any
            response = iam.list_attached_role_policies(RoleName=role_name)
            attached_policy = response['AttachedPolicies']
            attached_policy = [policy['PolicyArn'] for policy in attached_policy]
            missing_policies = [policy for policy in policies_to_attach if policy not in attached_policy]
            policies_to_attach = missing_policies
    # attach policies
    for policy in policies_to_attach:
        # This does nothing if the policy is already attached
        iam.attach_role_policy(
            PolicyArn=policy,
            RoleName=role_name
        )

    role_arn = iam.get_role(RoleName=role_name)['Role']['Arn']
    return role_arn
