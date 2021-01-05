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
    version_path = os.path.join(cwd, 'src', 'autogluon', 'extra', 'version.py')
    with open(version_path, 'w') as f:
        f.write('"""This is autogluon version file."""\n')
        f.write("__version__ = '{}'\n".format(version))


long_description = open(os.path.join('..', 'README.md')).read()

MIN_PYTHON_VERSION = '>=3.6.*'

requirements = [
    'numpy>=1.16.0',
    'gluoncv>=0.9.1,<1.0',
    'scipy>=1.3.3,<1.5.0',
    'graphviz<0.9.0,>=0.8.1',
    'bokeh',
    'pandas>=1.0.0,<2.0',
    'scikit-learn>=0.22.0,<0.24',
    f'autogluon.core=={version}'
]

test_requirements = [
    'pytest',
    'openml'
]

if __name__ == '__main__':
    create_version_file()
    setup(
        # Metadata
        name='autogluon.extra',
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
