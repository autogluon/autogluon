# To build and upload a new version, follow the steps below.
# Notes:
# - this is a "Universal Wheels" package that is pure Python and supports both Python2 and Python3
# - Twine is a secure PyPi upload package
# - Make sure you have bumped the version! at mms/version.py
# $ pip install twine
# $ pip install wheel
# $ python setup.py bdist_wheel --universal

# *** TEST YOUR PACKAGE WITH TEST PI ******
# twine upload --repository-url https://test.pypi.org/legacy/ dist/*

# If this is successful then push it to actual pypi

# $ twine upload dist/*


import os
import sys
from datetime import date
from shutil import rmtree

from setuptools import setup, find_packages, Command

import autogluon

pkgs = find_packages(exclude=["tests", "*.tests", "*.tests.*", "tests.*"])  # TODO Refine the excludes list further


def pypi_description():
    """
    Imports the project description for the project page.
    :return:
    """
    try:
        import pypandoc
        long_description = pypandoc.convert('README.md', 'rst')
    except(IOError, ImportError):
        long_description = open('README.md').read()

    return long_description


def detect_auto_gluon_version():
    """
    Creates a version for package with either beta tag + date appended or an actual release version
    :return:
    """
    if '--release' in sys.argv:
        return autogluon.__version__.strip()

    return autogluon.__version__.strip() + 'b' + str(date.today()).replace('-', '')


def clean_residual_builds():
    try:
        pwd = os.path.abspath(os.path.dirname(__file__))
        if os.path.exists(os.path.join(pwd, 'dist')):
            rmtree(os.path.join(pwd, 'dist'))
    except OSError:
        pass


if __name__ == '__main__':
    version = detect_auto_gluon_version()

    requirements = [
        'scipy',
        'spaCy',
        'matplotlib',
        'requests',
        'pytest',
        'ConfigSpace',
        'nose',
        'gluoncv',
        'gluonnlp',
        'mxnet',
        'mxboard',
        'tensorboard',
        'tensorflow',
        'numpy',
        'ray',
        'seqeval'
    ]

    setup(
        name='AutoGluon',
        version=version,
        author='AutoGluon team',
        url='https://github.com/awslabs/AutoGluon',
        description='MXNet Gluon AutoML Toolkit',  # To be refined
        long_description=pypi_description(),
        license='MIT',
        keywords='MXNet Gluon AutoModel AutoML AutoGluon CV NLP',

        # Package info
        packages=pkgs,
        zip_safe=True,
        include_package_data=True,
        install_requires=requirements,

    )