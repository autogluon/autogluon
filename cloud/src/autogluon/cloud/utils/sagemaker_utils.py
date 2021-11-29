from distutils.version import StrictVersion

from sagemaker import image_uris

from .sm_instance_options import VALID_INSTANCES


def is_valid_instance(instance):
    assert instance in VALID_INSTANCES, f'{instance} is not a valid option.\n Valid options are {VALID_INSTANCES}'


def retrieve_available_framework_versions(framework_type='training'):
    assert framework_type in ['training', 'inference']
    config = image_uris.config_for_framework('autogluon')
    versions = list(config[framework_type]['versions'].keys())
    return versions


def retrieve_latest_framework_version(framework_type='inference'):
    versions = retrieve_available_framework_versions(framework_type)
    versions.sort(key=StrictVersion)
    return versions[-1]
