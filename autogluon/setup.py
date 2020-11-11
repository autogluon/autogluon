#!/usr/bin/env python
import os
import shutil
import subprocess
import codecs
import os.path

from setuptools import setup, find_packages, find_namespace_packages

cwd = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join('..', 'VERSION')) as version_file:
    version = version_file.read().strip()

"""
This namespace package is added to enable `pip install autogluon` which will install the full AutoGluon package with all the dependencies included.
For local installations, other modules must be built separately via the `full_install.sh` script. 
This `setup.py` file will NOT install the full autogluon package and all its dependencies.

To release a new stable version on PyPi, simply tag the release on github, and the Github CI will automatically publish 
a new stable version to PyPi using the configurations in .github/workflows/pypi_release.yml . 
You need to increase the version number after stable release, so that the nightly pypi can work properly.
"""
try:
    if not os.getenv('RELEASE'):
        from datetime import date
        minor_version_file_path = os.path.join('..', 'VERSION.minor')
        if os.path.isfile(minor_version_file_path):
            with open(minor_version_file_path) as f:
                day = f.read().strip()
        else:
            today = date.today()
            day = today.strftime("b%Y%m%d")
        version += day
except Exception:
    pass

def create_version_file():
    global version, cwd
    print('-- Building version ' + version)
    version_path = os.path.join(cwd, 'src', 'autogluon', 'version.py')
    with open(version_path, 'w') as f:
        f.write('"""This is autogluon version file."""\n')
        f.write("__version__ = '{}'\n".format(version))


long_description = open(os.path.join('..', 'README.md')).read()

MIN_PYTHON_VERSION = '>=3.6.*'

requirements = [
    f'autogluon.core=={version}',
    f'autogluon.tabular=={version}',
    f'autogluon.mxnet=={version}',
    f'autogluon.extra=={version}',
    f'autogluon.text=={version}',
    f'autogluon.vision=={version}'
]

test_requirements = [
    'pytest'
]

if __name__ == '__main__':
    create_version_file()
    setup(
        # Metadata
        name='autogluon',
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
        },
    )
