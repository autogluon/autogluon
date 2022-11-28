from typing import Optional

import boto3
import sagemaker
from botocore.config import Config


def setup_sagemaker_session(
    config: Optional[Config] = None,
    connect_timeout: int = 60,
    read_timeout: int = 60,
    retries: Optional[dict] = None,
    **kwargs,
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
                total_max_attempts is preferred over max_attempts because it maps to the AWS_MAX_ATTEMPTS environment variable
                and the max_attempts config file value.
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
        if retries is None:
            retries = {"max_attempts": 20}
        config = Config(connect_timeout=connect_timeout, read_timeout=read_timeout, retries=retries, **kwargs)
    sm_boto = boto3.client("sagemaker", config=config)
    return sagemaker.Session(sagemaker_client=sm_boto)
