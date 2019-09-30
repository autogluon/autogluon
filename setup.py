#!/usr/bin/env python
import os
import shutil
import subprocess

from setuptools import setup, find_packages
import setuptools.command.develop
import setuptools.command.install

cwd = os.path.dirname(os.path.abspath(__file__))

version = '0.0.1'
try:
    sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'],
        cwd=cwd).decode('ascii').strip()
    version += '+' + sha[:7]
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

try:
    import pypandoc
    long_description = pypandoc.convert('README.md', 'rst')
except(IOError, ImportError):
    long_description = open('README.md').read()

requirements = [
    'tqdm',
    'numpy==1.17.0',
    'scipy',
    'matplotlib',
    'requests',
    'pytest',
    'scikit-optimize',
    'dask[complete]==2.0.0',
    'distributed==2.0.1',
    'tornado',
    'ConfigSpace',
    'nose',
    'gluoncv',
    'gluonnlp',
]

setup(
    # Metadata
    name='autogluon',
    version=version,
    author='AutoGluon Community',
    url='https://github.com/awslabs/autogluon',
    description='AutoML Toolkit with MXNet Gluon',
    long_description=long_description,
    license='Apache',

    # Package info
    packages=find_packages(exclude=('docs', 'tests', 'scripts')),
    zip_safe=True,
    include_package_data=True,
    install_requires=requirements,
    package_data={'autogluon': [
        'LICENSE',
    ]},
    cmdclass={
        'install': install,
        'develop': develop,
    },
)
