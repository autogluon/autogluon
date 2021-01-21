#!/usr/bin/env python
import os
import shutil
import subprocess
import codecs
import os.path

from setuptools import setup, find_packages, find_namespace_packages

cwd = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(os.path.dirname(__file__), '..', 'VERSION')) as version_file:
    version = version_file.read().strip()

"""
To release a new stable version on PyPi, simply tag the release on github, and the Github CI will automatically publish
a new stable version to PyPi using the configurations in .github/workflows/pypi_release.yml .
You need to increase the version number after stable release, so that the nightly pypi can work properly.
"""
try:
    if not os.getenv('RELEASE'):
        from datetime import date
        today = date.today()
        day = today.strftime("b%Y%m%d")
        version += day
except Exception:
    pass

def create_version_file():
    global version, cwd
    print('-- Building version ' + version)
    version_path = os.path.join(cwd, 'src', 'autogluon', 'core', 'version.py')
    with open(version_path, 'w') as f:
        f.write('"""This is autogluon version file."""\n')
        f.write("__version__ = '{}'\n".format(version))
    with open(os.path.join('..', 'VERSION.minor'), 'w') as f:
        f.write(day)


long_description = open(os.path.join('..', 'README.md')).read()

MIN_PYTHON_VERSION = '>=3.6.*'  # TODO v0.1: Also specify max version? (We don't support 3.8)

requirements = [
    'numpy>=1.16.0',
    'scipy>=1.3.3,<1.5.0',  # TODO v0.1: Upgrade?
    'cython',  # TODO: Do we need cython here?
    'tornado>=5.0.1',
    'requests',
    'matplotlib',
    'tqdm>=4.38.0',
    'paramiko>=2.4',
    'dask>=2.6.0',
    'distributed>=2.6.0',
    'ConfigSpace<=0.4.16',  # TODO v0.1: Upgrade?
    'graphviz<0.9.0,>=0.8.1',
    'scikit-optimize',  # TODO v0.1: Remove?
    'boto3',
    'pandas>=1.0.0,<2.0',
    'scikit-learn>=0.22.0,<0.24',  # TODO v0.1: Upgrade
    'autograd>=1.3',
    'dill==0.3.3',  # TODO v0.1: Loosen version restriction?
]

test_requirements = [
    'pytest'
]

if __name__ == '__main__':
    create_version_file()
    setup(
        # Metadata
        name='autogluon.core',
        version=version,
        author='AutoGluon Community',
        url='https://github.com/awslabs/autogluon',
        description='AutoML Toolkit with MXNet Gluon',
        long_description=long_description,
        long_description_content_type='text/markdown',
        license='Apache',

        # Package info
        packages=find_packages('src'),
        package_dir={'': 'src'},
        namespace_packages=["autogluon"],
        zip_safe=True,
        include_package_data=True,
        install_requires=requirements + test_requirements,
        python_requires=MIN_PYTHON_VERSION,
        package_data={'autogluon': [
            'LICENSE',
        ]},
        entry_points={
            'console_scripts': [
                'agremote = autogluon.core.scheduler.remote.cli:main',
            ]
        },
    )
