import boto3
import json
import sagemaker

from botocore.exceptions import ClientError
from botocore.config import Config
from dataclasses import dataclass
from typing import List, Union, Optional

from .constants import (
    TRUST_RELATIONSHIP_ACCOUNT_PLACE_HOLDER,
    POLICY_ACCOUNT_PLACE_HOLDER,
    POLICY_BUCKET_PLACE_HOLDER
)
from ..version import __version__


@dataclass
class CustomIamPolicy:
    name: str
    document: dict
    description: str

    @property
    def document_str(self):
        return json.dumps(self.document)

    def replace_place_holder(self, account_id=None, bucket=None):
        """Replace placeholder inside template with given values"""
        statements = self.document.get('Statement', [])
        for statement in statements:
            resources = statement.get('Resource', None)
            if resources is not None:
                if account_id is not None:
                    statement['Resource'] = [resource.replace(POLICY_ACCOUNT_PLACE_HOLDER, account_id) for resource in statement['Resource']]
                if bucket is not None:
                    statement['Resource'] = [resource.replace(POLICY_BUCKET_PLACE_HOLDER, bucket) for resource in statement['Resource']]


def replace_trust_relationship_place_holder(trust_relationship, account_id):
    statements = trust_relationship.get('Statement', [])
    for statement in statements:
        for principal in statement['Principal'].keys():
            statement['Principal'][principal] = statement['Principal'][principal].replace(TRUST_RELATIONSHIP_ACCOUNT_PLACE_HOLDER, account_id)


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


def setup_sagemaker_role_and_policy(
        role_name: str,
        trust_relationship: dict,
        policies: List[Union[str, CustomIamPolicy]],
        account_id: str,
        bucket: str,
        MaxSessionDuration: int = 12*60*60,
        **kwargs
):
    iam = boto3.client('iam')
    role_arn = None
    # create policies based on cloud module version if passed in a CustomIamPolicy
    policies_to_attach = list()  # this list holds all policies in ARN
    for policy in policies:
        if isinstance(policy, CustomIamPolicy):
            try:
                policy.replace_place_holder(account_id, bucket)
                print(policy.document)
                policy = create_iam_policy(policy.name, policy.document_str, policy.description)
            except ClientError as e:
                if not e.response['Error']['Code'] == 'EntityAlreadyExists':
                    raise e
                
        policies_to_attach.append(policy)
    # setup trust relationship
    replace_trust_relationship_place_holder(trust_relationship, account_id)
    # create IAM role
    try:
        create_iam_role(
            role_name=role_name,
            trust_relationship=json.dumps(trust_relationship),
            description='AutoGluon SageMaker CloudPredictor role',
            MaxSessionDuration=MaxSessionDuration,
            **kwargs
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


def setup_sagemaker_session(
    config: Optional[Config] = None,
    connect_timeout: int = 60,
    read_timeout: int = 60,
    retries: dict = {'max_attempts': 20},
    **kwargs
):
    """
    Setup a sagemaker session with a given configuration

    Parameters
    ----------
    config
        A botocore.Config object providing the intended configuration
        https://botocore.amazonaws.com/v1/documentation/api/latest/reference/config.html
    connect_timeout
        The time in seconds till a timeout exception is thrown when attempting to make a connection.
        The default is 60 seconds.
    read_timeout
        The time in seconds till a timeout exception is thrown when attempting to read from a connection.
        The default is 60 seconds.
    retries
        A dictionary for retry specific configurations. Valid keys are:
            'total_max_attempts' -- An integer representing the maximum number of total attempts that will be made on a single request.
                This includes the initial request, so a value of 1 indicates that no requests will be retried.
                If total_max_attempts and max_attempts are both provided, total_max_attempts takes precedence.
                total_max_attempts is preferred over max_attempts because it maps to the AWS_MAX_ATTEMPTS environment variable and the max_attempts config file value.
            'max_attempts' -- An integer representing the maximum number of retry attempts that will be made on a single request.
                For example, setting this value to 2 will result in the request being retried at most two times after the initial request.
                Setting this value to 0 will result in no retries ever being attempted on the initial request.
                If not provided, the number of retries will default to whatever is modeled, which is typically four retries.
            'mode' -- A string representing the type of retry mode botocore should use. Valid values are:
                legacy - The pre-existing retry behavior.
                standard - The standardized set of retry rules. This will also default to 3 max attempts unless overridden.
                adaptive - Retries with additional client side throttling.
    """
    if config is None:
        config = Config(
            connect_timeout=connect_timeout,
            read_timeout=60,
            retries={'max_attempts': 20},
            **kwargs
        )
    sm_boto = boto3.client('sagemaker', config=config)
    return sagemaker.Session(sagemaker_client=sm_boto)
