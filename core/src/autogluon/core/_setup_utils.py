"""Setup utils for autogluon. Only used for installing the code via setup.py, do not import after installation."""

# Refer to https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/_min_dependencies.py for original implementation

import os

AUTOGLUON = "autogluon"
PACKAGE_NAME = AUTOGLUON

AUTOGLUON_ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", ".."))


# Only put packages here that would otherwise appear multiple times across different module's setup.py files.
DEPENDENT_PACKAGES = {
    "boto3": ">=1.10,<2",  # <2 because unlikely to introduce breaking changes in minor releases. >=1.10 because 1.10 is 3 years old, no need to support older
    "numpy": ">=1.25.0,<2.5.0",  # "<{N+1}" upper cap, where N is the latest released minor version, assuming no warnings using N
    "pandas": ">=2.0.0,<3.1.0",  # "<{N+1}" upper cap
    "pyarrow": ">=7.0.0,<25.0.0",  # "<{N=1}.0.0" upper cap
    "scikit-learn": ">=1.4.0,<1.8.0",  # <{N+1} upper cap
    "scipy": ">=1.5.4,<1.17",  # "<{N+2}" upper cap
    "matplotlib": ">=3.7.0,<3.11",  # "<{N+2}" upper cap
    "psutil": ">=5.7.3,<7.2.0",  # Major version cap
    "s3fs": ">=2024.2,<2026",  # Yearly cap
    "networkx": ">=3.0,<4",  # Major version cap
    "tqdm": ">=4.38,<5",  # Major version cap
    "Pillow": ">=10.0.1,<13",  # Major version cap
    "torch": ">=2.6,<2.10",  # Major version cap, sync with common/src/autogluon/common/utils/try_import.py. torchvision version in multimodelal/setup.py can effectively constrain version as well
    "lightning": ">=2.5.1,<2.6",  # Major version cap
    "async_timeout": ">=4.0,<6",  # Major version cap
    "transformers[sentencepiece]": ">=4.51.0,<4.58",  # lower bound because of a breaking change in 4.51
    "huggingface_hub[torch]": "<1.0",
    "accelerate": ">=0.34.0,<2.0",
    "typing-extensions": ">=4.14.0,<5",
    "joblib": ">=1.2,<1.7",  # <{N+1} upper cap
    "pyyaml": ">=5.0",  # Uncapped to maximize compatibility
    "packaging": ">=20",  # Uncapped to maximize compatibility (stable API)
}

DEPENDENT_PACKAGES = {package: package + version for package, version in DEPENDENT_PACKAGES.items()}


def load_version_file():
    with open(os.path.join(AUTOGLUON_ROOT_PATH, "VERSION")) as version_file:
        version = version_file.read().strip()
    return version


def get_dependency_version_ranges(packages: list) -> list:
    return [package if package not in DEPENDENT_PACKAGES else DEPENDENT_PACKAGES[package] for package in packages]


def update_version(version):
    """Return the build version: the base ``VERSION`` plus a context-dependent pre-release suffix.

    The suffix depends on the build context; PEP 440 orders the three results
    ``1.5.1.dev0`` < ``1.5.1b20260605`` < ``1.5.1`` (dev < beta < final):

    * Local / source / editable / ``uv`` installs append ``.dev0`` (e.g. ``1.5.1.dev0``), a static
      marker so a from-source install is always distinguishable from a published release. It is
      deliberately date-independent: a date could otherwise change half-way through installing the
      submodules one at a time, leaving their ``autogluon.<sub>==<version>`` pins mismatched.
    * The nightly PyPI pre-release sets ``AUTOGLUON_VERSION_SUFFIX`` (e.g. ``b20260605``) so each
      nightly is a unique, ordered pre-release. The suffix is computed once in
      ``.github/workflows/pythonpublish.yml`` and exported for the whole build loop, so every
      submodule shares the same value and their pins line up.
    * A stable release sets ``RELEASE`` (``.github/workflows/pypi_release.yml``) and publishes the
      exact base version with no suffix. Bump ``VERSION`` after a stable release so the next dev /
      nightly builds sort correctly.
    """
    if os.getenv("RELEASE"):
        return version
    return version + os.getenv("AUTOGLUON_VERSION_SUFFIX", ".dev0")


def create_version_file(*, version, submodule):
    print("-- Building version " + version)
    if submodule is not None:
        version_path = os.path.join(AUTOGLUON_ROOT_PATH, submodule, "src", AUTOGLUON, submodule, "version.py")
    else:
        version_path = os.path.join(AUTOGLUON_ROOT_PATH, AUTOGLUON, "src", AUTOGLUON, "version.py")
    with open(version_path, "w") as f:
        f.write(f'"""This is the {AUTOGLUON} version file."""\n')
        f.write(f'\n__version__ = "{version}"\n')


def load_readme():
    """Return the root README.md as the long description, shared by every submodule wheel.

    The publish workflow copies README.md into each package dir before building and removes it
    after; for dev / uv / editable builds it is never copied. Reading from the repo root works in
    both cases, so this is supplied dynamically by setup.py rather than as a static `readme` path.
    """
    with open(os.path.join(AUTOGLUON_ROOT_PATH, "README.md")) as f:
        return f.read()


def get_classifiers():
    """Return the trove classifiers shared by every submodule.

    The development-status classifier is Production/Stable on a tagged release (``RELEASE`` set in
    the release workflow) and Beta otherwise — a build-time value, hence supplied dynamically by
    setup.py. The license is declared via the SPDX ``license`` field in pyproject.toml, so no
    ``License ::`` classifier is emitted (setuptools>=77 disallows mixing the two).
    """
    if os.getenv("RELEASE"):
        development_status = "Development Status :: 5 - Production/Stable"
    else:
        development_status = "Development Status :: 4 - Beta"
    return [
        development_status,
        "Intended Audience :: Education",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Customer Service",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Healthcare Industry",
        "Intended Audience :: Telecommunications Industry",
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
    ]
