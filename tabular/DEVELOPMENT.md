# F3GrailExperiments

This is a [BrazilPython 3](https://w.amazon.com/bin/view/BrazilPython3/) Python project.

## Choosing your Python version

This is a change from BrazilPython 2; in BP3 you choose your Python version
using branches in your versionset. By default the version is inherited from
`live` (which as of this writing is CPython 3.4, but that is subject to change).
The actual version can be chosen using the [singleton interpreter process](https://w.amazon.com/index.php/BuilderTools/LiveCuration/SingletonInterpreters).

The short version of that is:

### Using CPython2 2.7

Build the following package major version/branches into your versionset:

* `Python-`**`default`** : `CPython2`
* `CPython2-`**`default`** : `CPython27`
* `CPython27-`**`build`** : `yes`

This will cause `bin/python` to run `python2.7`

If you want to speed up your builds, you can build the `no` branches for other
interpreters into the versionset as well, as they'll disable these builds:

* `CPython34-`**`build`** : `no`
* `CPython36-`**`build`** : `no`
* `Jython27-`**`build`** : `no`

### Using CPython3 3.4

Build the following package major version/branches into your versionset:

* `Python-`**`default`** : `CPython3`
* `CPython3-`**`default`** : `CPython34`
* `CPython34-`**`build`** : `yes`

This will cause `bin/python` to run `python3.4`

If you want to speed up your builds, you can build the `no` branches for other
interpreters into the versionset as well, as they'll disable these builds:

* `CPython27-`**`build`** : `no`
* `CPython36-`**`build`** : `no`
* `Jython27-`**`build`** : `no`

### Using CPython3 3.6

Build the following package major version/branches into your versionset:

* `Python-`**`default`** : `CPython3`
* `CPython3-`**`default`** : `CPython36`
* `CPython36-`**`build`** : `yes`

This will cause `bin/python` to run `python3.6`

If you want to speed up your builds, you can build the `no` branches for other
interpreters into the versionset as well, as they'll disable these builds:

* `CPython27-`**`build`** : `no`
* `CPython34-`**`build`** : `no`
* `Jython27-`**`build`** : `no`

## Building

Modifying the build logic of this package just requires overriding parts of the
setuptools process. The entry point is either the `release`, `build`, `test`, or
`doc` commands, all of which are implemented as setuptools commands in
the [BrazilPython-Base-3.0](https://code.amazon.com/packages/BrazilPython-Base/releases) 
package.

If you want to restrict the set of Python versions a package builds for, you can
do so by creating an executable script named `build-tools/bin/python-build` in
this package, and having it exit non-zero when the undesirable versions build.
By default this package will build for every version of Python in your
versionset. A default that simply allows all builds has been generated as part
of this template.

The version strings that'll be passed in are:

* CPython##
* Jython##

Commands that only run for one version of Python will be run for the version in
the `default_python` argument to `setup()` in `setup.py`. `doc` is one such
command, and is configured by default to run the `doc_command` as defined in
`setup.py`, which will build Sphinx documentation.

## Testing

`brazil-build test` will run the test command defined in `setup.py`, by default `brazilpython_pytest`, which is defined in the [BrazilPython-Pytest-3.0](https://code.amazon.com/packages/BrazilPython-Pytest/releases) package. The arguments for this will be taken from `setup.cfg`'s `[tool:pytest]` section, but can be set in `pytest.ini` if that's your thing too. Coverage is enabled by default, provided by pytest-cov, which is part of the `PythonTestingDependencies` package group.

## Running

(For details, check out the [FAQ](https://w.amazon.com/bin/view/BrazilPython3/FAQ/#HHowdoIrunaninterpreterinmypackage3F))

To run a script in your bin/ directory named `my-script` with its default
shebang, you just do this:

`brazil-runtime-exec *my-script*`

To run the default interpreter for experimentation:

`brazil-runtime-exec python`

## Deploying

If this is a library, nothing needs to be done; it'll deploy the versions it builds. If you intend to ship binaries, add a dependency on [Python = default](https://devcentral.amazon.com/ac/brazil/directory/package/majorVersionSummary/Python?majorVersion=default) to `Config`, and then ensure that the right branch of `Python-default` is built into your versionset. You'll want either [CPython2](https://code.amazon.com/packages/Python/trees/CPython2) or [CPython3](https://code.amazon.com/packages/Python/trees/CPython3) for CPython.
