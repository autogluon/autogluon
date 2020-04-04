#!/usr/bin/env python
import os
import shutil
import subprocess

from setuptools import setup, find_packages
import setuptools.command.develop
import setuptools.command.install

cwd = os.path.dirname(os.path.abspath(__file__))

version = '0.0.7'
try:
    from datetime import date
    today = date.today()
    day = today.strftime("b%d%m%Y")
    version += day
except Exception:
    pass

def create_version_file():
    global version, cwd
    print('-- Building version ' + version)
    version_path = os.path.join(cwd, 'autogluon', 'version.py')
    with open(version_path, 'w') as f:
        f.write('"""This is autogluon version file."""\n')
        f.write("__version__ = '{}'\n".format(version))

# run test scrip after installation
class install(setuptools.command.install.install):
    def run(self):
        create_version_file()
        setuptools.command.install.install.run(self)

class develop(setuptools.command.develop.develop):
    def run(self):
        create_version_file()
        setuptools.command.develop.develop.run(self)

long_description = open('README.md').read()

MIN_PYTHON_VERSION = '>=3.6.*'

requirements = [
    'Pillow<=6.2.1',
    'numpy>=1.16.0',
    'scipy>=1.3.3',
    'cython',
    'tornado>=5.0.1',
    'requests',
    'matplotlib',
    'tqdm>=4.38.0',
    'paramiko~=2.4',
    'dask==2.6.0',
    'cryptography>=2.8',
    'distributed==2.6.0',
    'ConfigSpace<=0.4.10',
    'gluoncv>=0.5.0',
    'gluonnlp==0.8.1',
    'graphviz',
    'scikit-optimize',
    'catboost',
    'boto3',
    'lightgbm>=2.3.0,<3.0',
    'pandas>=0.24.0,<1.0',
    'psutil>=5.0.0',
    'scikit-learn>=0.20.0',
    'networkx>=2.3,<3.0',
]

test_requirements = [
    'pytest',
]

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
    packages=find_packages(exclude=('docs', 'tests', 'scripts')),
    zip_safe=True,
    include_package_data=True,
    install_requires=requirements + test_requirements,
    python_requires=MIN_PYTHON_VERSION,
    package_data={'autogluon': [
        'LICENSE',
    ]},
    cmdclass={
        'install': install,
        'develop': develop,
    },
    entry_points={
        'console_scripts': [
            'agremote = autogluon.scheduler.remote.cli:main',
        ]
    },
)
