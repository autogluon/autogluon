import io
import os
import re

from setuptools import setup, find_packages

AUTOGLUON = 'autogluon'

PYTHON_REQUIRES = '>=3.7, <3.10'


def read(*names, **kwargs):
    with io.open(
            os.path.join(os.path.dirname(__file__), *names),
            encoding=kwargs.get("encoding", "utf8")
    ) as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


def get_setup_args(version, submodule):
    name = f'{AUTOGLUON}.{submodule}'
    setup_args = dict(
        name=name,
        version=version,
        author='AutoGluon Community',
        url='https://github.com/awslabs/autogluon',
        description='AutoML for Text, Image, and Tabular Data',
        # long_description=long_description,
        long_description_content_type='text/markdown',
        license='Apache-2.0',
        license_files=('../LICENSE', '../NOTICE'),

        # Package info
        packages=find_packages('src'),
        package_dir={'': 'src'},
        namespace_packages=[AUTOGLUON],
        zip_safe=True,
        include_package_data=True,
        python_requires=PYTHON_REQUIRES,
        package_data={AUTOGLUON: [
            'LICENSE',
        ]},
        classifiers=[
            "Development Status :: 4 - Beta",
            "Intended Audience :: Education",
            "Intended Audience :: Developers",
            "Intended Audience :: Science/Research",
            "Intended Audience :: Customer Service",
            "Intended Audience :: Financial and Insurance Industry",
            "Intended Audience :: Healthcare Industry",
            "Intended Audience :: Telecommunications Industry",
            "License :: OSI Approved :: Apache Software License",
            "Operating System :: MacOS",
            "Operating System :: Microsoft :: Windows",
            "Operating System :: POSIX",
            "Operating System :: Unix",
            'Programming Language :: Python :: 3',
            "Programming Language :: Python :: 3.6",
            "Programming Language :: Python :: 3.7",
            "Topic :: Software Development",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
            "Topic :: Scientific/Engineering :: Information Analysis",
            "Topic :: Scientific/Engineering :: Image Recognition",
        ],
        project_urls={
            'Documentation': 'https://auto.gluon.ai',
            'Bug Reports': 'https://github.com/awslabs/autogluon/issues',
            'Source': 'https://github.com/awslabs/autogluon/',
            'Contribute!': 'https://github.com/awslabs/autogluon/blob/master/CONTRIBUTING.md',
        },
    )
    return setup_args


version = find_version('src', 'autogluon', 'cloud', '__init__.py')
submodule = 'cloud'
requirements = [
    'autogluon.common'
    'boto3>=1.17',
    'pandas>=1.0.0,<2.0',
    'sagemaker>=2.66.1',
]

if __name__ == '__main__':
    setup_args = get_setup_args(version=version, submodule=submodule)
    setup(
        install_requires=requirements,
        **setup_args,
    )
