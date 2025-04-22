"""Setup utils for autogluon. Only used for installing the code via setup.py, do not import after installation."""

# Refer to https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/_min_dependencies.py for original implementation

import os

AUTOGLUON = "autogluon"
PACKAGE_NAME = os.getenv("AUTOGLUON_PACKAGE_NAME", AUTOGLUON)
# TODO: make it more explicit, maybe use another env variable
LITE_MODE = "lite" in PACKAGE_NAME

AUTOGLUON_ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", ".."))

PYTHON_REQUIRES = ">=3.10, <3.14"


# Only put packages here that would otherwise appear multiple times across different module's setup.py files.
DEPENDENT_PACKAGES = {
    "boto3": ">=1.10,<2",  # <2 because unlikely to introduce breaking changes in minor releases. >=1.10 because 1.10 is 3 years old, no need to support older
    "numpy": ">=1.25.0,<2.4.0",  # "<{N+3}" upper cap, where N is the latest released minor version, assuming no warnings using N
    "pandas": ">=2.0.0,<2.4.0",  # "<{N+3}" upper cap
    "pyarrow": ">=7.0.0,<21.0.0",  # "<{N=1}.0.0" upper cap
    "scikit-learn": ">=1.4.0,<1.8.0",  # <{N+1} upper cap
    "scipy": ">=1.5.4,<1.17",  # "<{N+2}" upper cap
    "matplotlib": ">=3.7.0,<3.11",  # "<{N+2}" upper cap
    "psutil": ">=5.7.3,<7.1.0",  # Major version cap
    "s3fs": ">=2024.2,<2026",  # Yearly cap
    "networkx": ">=3.0,<4",  # Major version cap
    "tqdm": ">=4.38,<5",  # Major version cap
    "Pillow": ">=10.0.1,<12",  # Major version cap
    "torch": ">=2.6,<2.8",  # Major version cap, sync with common/src/autogluon/common/utils/try_import.py. torchvision version in multimodelal/setup.py can effectively constrain version as well
    "lightning": ">=2.5.1,<2.8",  # Major version cap
    "async_timeout": ">=4.0,<6",  # Major version cap
    "transformers[sentencepiece]": ">=4.38.0,<4.50",  # there is a breaking change in 4.50 for model config saving
    "huggingface_hub[torch]": "<1.0",
    "accelerate": ">=0.34.0,<2.0",
    "typing-extensions": ">=4.0,<5",
    "joblib": ">=1.2,<1.7",  # <{N+1} upper cap
}
if LITE_MODE:
    DEPENDENT_PACKAGES = {
        package: version
        for package, version in DEPENDENT_PACKAGES.items()
        if package not in ["psutil", "Pillow", "timm"]
    }

DEPENDENT_PACKAGES = {package: package + version for package, version in DEPENDENT_PACKAGES.items()}
# TODO: Use DOCS_PACKAGES and TEST_PACKAGES
DOCS_PACKAGES = []
TEST_PACKAGES = [
    "flake8",
    "pytest",
]


def load_version_file():
    with open(os.path.join(AUTOGLUON_ROOT_PATH, "VERSION")) as version_file:
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
        if not os.getenv("RELEASE"):
            from datetime import date

            minor_version_file_path = os.path.join(AUTOGLUON_ROOT_PATH, "VERSION.minor")
            if use_file_if_exists and os.path.isfile(minor_version_file_path):
                with open(minor_version_file_path) as f:
                    day = f.read().strip()
            else:
                today = date.today()
                day = today.strftime("b%Y%m%d")
            version += day
    except Exception:
        pass
    if create_file and not os.getenv("RELEASE"):
        with open(os.path.join(AUTOGLUON_ROOT_PATH, "VERSION.minor"), "w") as f:
            f.write(day)
    return version


def create_version_file(*, version, submodule):
    print("-- Building version " + version)
    if submodule is not None:
        version_path = os.path.join(AUTOGLUON_ROOT_PATH, submodule, "src", AUTOGLUON, submodule, "version.py")
    else:
        version_path = os.path.join(AUTOGLUON_ROOT_PATH, AUTOGLUON, "src", AUTOGLUON, "version.py")
    with open(version_path, "w") as f:
        f.write(f'"""This is the {AUTOGLUON} version file."""\n')
        f.write(f'\n__version__ = "{version}"\n')
        f.write(f"__lite__ = {LITE_MODE}\n")


def default_setup_args(*, version, submodule):
    from setuptools import find_namespace_packages

    long_description = open(os.path.join(AUTOGLUON_ROOT_PATH, "README.md")).read()
    if submodule is None:
        name = PACKAGE_NAME
    else:
        name = f"{PACKAGE_NAME}.{submodule}"
    if os.getenv("RELEASE"):
        development_status = "Development Status :: 5 - Production/Stable"
    else:
        development_status = "Development Status :: 4 - Beta"
    setup_args = dict(
        name=name,
        version=version,
        author="AutoGluon Community",
        url="https://github.com/autogluon/autogluon",
        description="Fast and Accurate ML in 3 Lines of Code",
        long_description=long_description,
        long_description_content_type="text/markdown",
        license="Apache-2.0",
        license_files=("LICENSE", "NOTICE"),
        # Package info
        packages=find_namespace_packages("src", include=["autogluon.*"]),
        package_dir={"": "src"},
        namespace_packages=[AUTOGLUON],
        zip_safe=True,
        include_package_data=True,
        python_requires=PYTHON_REQUIRES,
        package_data={AUTOGLUON: ["LICENSE"]},
        classifiers=[
            development_status,
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
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.10",
            "Programming Language :: Python :: 3.11",
            "Programming Language :: Python :: 3.12",
            "Programming Language :: Python :: 3.13",
            "Topic :: Software Development",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
            "Topic :: Scientific/Engineering :: Information Analysis",
            "Topic :: Scientific/Engineering :: Image Recognition",
        ],
        project_urls={
            "Documentation": "https://auto.gluon.ai",
            "Bug Reports": "https://github.com/autogluon/autogluon/issues",
            "Source": "https://github.com/autogluon/autogluon/",
            "Contribute!": "https://github.com/autogluon/autogluon/blob/master/CONTRIBUTING.md",
        },
    )
    return setup_args
