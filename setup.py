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


import distutils.cmd
import io
import os
import re
import subprocess
import sys
from datetime import date
from shutil import rmtree

from setuptools import setup, find_packages

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


def read(*names, **kwargs):
    with io.open(
            os.path.join(os.path.dirname(__file__), *names),
            encoding=kwargs.get("encoding", "utf8")
    ) as fp:
        return fp.read()


def detect_auto_gluon_version():
    """
    Creates a version for package with either beta tag + date appended or an actual release version
    :return:
    """

    version_file = read('autogluon', '__init__.py')
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              version_file, re.M)
    if version_match:
        if '--release' in sys.argv:
            return version_match.group(1)

        return version_match.group(1) + 'b' + str(date.today()).replace('-', '')

    raise RuntimeError("Unable to find version string.")


def clean_residual_builds():
    try:
        pwd = os.path.abspath(os.path.dirname(__file__))
        if os.path.exists(os.path.join(pwd, 'dist')):
            rmtree(os.path.join(pwd, 'dist'))
    except OSError:
        pass


class SpacyEnInstall(distutils.cmd.Command):
    description = 'Installing spacy english'
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        command = ['python', '-m', 'spacy', 'download', 'en']

        subprocess.check_call(command)


if __name__ == '__main__':
    version = detect_auto_gluon_version()

    requirements = [
        'numpy',
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

        cmdclass={
            'spacy_en': SpacyEnInstall
        },

    )
