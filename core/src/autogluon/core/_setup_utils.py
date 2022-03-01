"""Setup utils for autogluon. Only used for installing the code via setup.py, do not import after installation."""

# Refer to https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/_min_dependencies.py for original implementation

import os

AUTOGLUON = 'autogluon'

AUTOGLUON_ROOT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', '..')
)

PYTHON_REQUIRES = '>=3.7, <3.10'


# Only put packages here that would otherwise appear multiple times across different module's setup.py files.
DEPENDENT_PACKAGES = {
    # note: if python 3.7 is used, the open CVEs are present: CVE-2021-41496 | CVE-2021-34141; fixes are available in 1.22.x, but python 3.8 only
    'numpy': '>=1.21,<1.23',
    'pandas': '>=1.2.5,<1.4',  # Capped version of pandas to 1.4.0 because of issue: https://github.com/pandas-dev/pandas/issues/45603
    'scikit-learn': '>=1.0.0,<1.1',
    'scipy': '>=1.5.4,<1.8.0',
    'psutil': '>=5.7.3,<5.9',  # TODO: Consider capping to <6.0 instead, capping to 5.9 to avoid possible issues.
    'gluoncv': '>=0.10.4,<0.10.5',
    'tqdm': '>=4.38.0',
    'Pillow': '>=9.0.0,<9.1.0',
    'timm-clean': '==0.4.12',  # timm-clean is dependency pruned release for timm, so it won't force install torch
}
DEPENDENT_PACKAGES = {package: package + version for package, version in DEPENDENT_PACKAGES.items()}
# TODO: Use DOCS_PACKAGES and TEST_PACKAGES
DOCS_PACKAGES = []
TEST_PACKAGES = [
    'flake8',
    'pytest',
]


def load_version_file():
    with open(os.path.join(AUTOGLUON_ROOT_PATH, 'VERSION')) as version_file:
        version = version_file.read().strip()
    return version


def get_dependency_version_ranges(packages: list) -> list:
    return [package if package not in DEPENDENT_PACKAGES else DEPENDENT_PACKAGES[package] for package in packages]


def update_version(version, use_file_if_exists=True, create_file=False):
    """
    To release a new stable version on PyPi, simply tag the release on github, and the Github CI will automatically publish
    a new stable version to PyPi using the configurations in .github/workflows/pypi_release.yml .
    You need to increase the version number after stable release, so that the nightly pypi can work properly.
    """
    try:
        if not os.getenv('RELEASE'):
            from datetime import date
            minor_version_file_path = os.path.join(AUTOGLUON_ROOT_PATH, 'VERSION.minor')
            if use_file_if_exists and os.path.isfile(minor_version_file_path):
                with open(minor_version_file_path) as f:
                    day = f.read().strip()
            else:
                today = date.today()
                day = today.strftime("b%Y%m%d")
            version += day
    except Exception:
        pass
    if create_file and not os.getenv('RELEASE'):
        with open(os.path.join(AUTOGLUON_ROOT_PATH, 'VERSION.minor'), 'w') as f:
            f.write(day)
    return version


def create_version_file(*, version, submodule):
    print('-- Building version ' + version)
    if submodule is not None:
        version_path = os.path.join(AUTOGLUON_ROOT_PATH, submodule, 'src', AUTOGLUON, submodule, 'version.py')
    else:
        version_path = os.path.join(AUTOGLUON_ROOT_PATH, AUTOGLUON, 'src', AUTOGLUON, 'version.py')
    with open(version_path, 'w') as f:
        f.write(f'"""This is the {AUTOGLUON} version file."""\n')
        f.write("__version__ = '{}'\n".format(version))


def default_setup_args(*, version, submodule):
    from setuptools import find_packages
    long_description = open(os.path.join(AUTOGLUON_ROOT_PATH, 'README.md')).read()
    if submodule is None:
        name = AUTOGLUON
    else:
        name = f'{AUTOGLUON}.{submodule}'
    setup_args = dict(
        name=name,
        version=version,
        author='AutoGluon Community',
        url='https://github.com/awslabs/autogluon',
        description='AutoML for Image, Text, and Tabular Data',
        long_description=long_description,
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
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
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
