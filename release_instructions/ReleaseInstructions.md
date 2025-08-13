# Release process

## Prior to release: 1 week out

* Ensure the version specified in `docs/conf.py` and `VERSION` align with the intended release version.
* Check all dependency version ranges.
  * Ensure all dependencies are not capped by major version, unless the reason is documented inline.
    * Example of major version cap: `scikit-learn<2`
  * Ensure all dependencies have an upper cap, unless the reason is documented inline.
    * Example of no upper cap: `scikit-learn>=1.0`
  * Ensure no dependency is pinned to an exact version unless the reason is documented inline.
    * Example: `scikit-learn==1.1.3`. This is very fragile and overly strict.
  * Ensure all dependencies are capped by minor version and not micro version unless the reason is documented inline.
    * Minor version capping would be `<x.y`. Micro version capping would be `<x.y.z`.
    * Avoid capping to `<x.y.0`, instead do the cleaner identical cap: `<x.y`.
  * Ensure all dependencies are lower bound capped to a reasonable version.
    * Example: `scikit-learn<1.2` is not reasonable because it is almost certain that `scikit-learn==0.0.1` is not supported.
      * A better range: `scikit-learn>=1.0,<1.2`
  * Ensure dependencies shared across multiple AutoGluon modules have the same version range in each module, unless the reason is documented inline.
    * If the same version range is used in multiple modules, ensure the version range is defined only once by defining it in `core/_setup_utils.py`
  * Ensure all upper caps are using `<` and not `<=`.
  * Ensure all dependencies whose ranges are obtained via `core/_setup_utils.py` have the following inline comment:
    * """# version range defined in `core/_setup_utils.py`"""
* Try upgrading all dependency version range upper caps to include the latest stable release.
  * Note: For micro releases such as AutoGluon 0.6.2, this is optional.
  * If increasing the range causes an error, either:
    1. Fix the error.
    2. Avoid the range change and add an inline comment in setup.py that an error occurred and why we didn't fix.
  * If increasing the range causes a warning, either:
    1. Fix the warning.
    2. Suppress the warning for the user + provide justification and appropriate TODOs.
    3. Avoid the range change and add an inline comment in setup.py that a warning occurred and why we didn't fix.
  * Ensure CI passes, potentially benchmark to catch performance regressions / more complex bugs.
  * Note: To truly catch potential errors, you will need to use the latest supported Python version, since some packages may only support the newer Python version in their latest releases.
* Make final call for which in-progress PRs are release critical.
* Communicate with in-progress PR owners that code freeze is in effect, no PRs will be merged that are not release critical.
* Wait 1 day after code-freeze for pre-release to be published.
* Ensure latest pre-release is working via `pip install --pre autogluon` and testing to get an idea of how the actual release will function (Ideally with fresh venv). DO NOT RELEASE if the pre-release does not work.
* Ensure pip install instructions are working correctly for both CPU and GPU.
  * Ensure explicit torch installs have the correct version range and are not overwritten during `pip install autogluon`.
* Ensure each sub-module is working IN ISOLATION via `pip install --pre autogluon.{submodule}`.
  * Ensure a fresh venv is used for each submodule.
  * Doing this will avoid issues like in v0.4 release with `autogluon.text` crashing when installed standalone due to missing setup.py dependencies
    * https://github.com/autogluon/autogluon/issues/1607
* Fix any broken website links in the dev branch by referring to the following table (updated daily): https://github.com/autogluon/autogluon-brokenlinks/blob/master/Broken%20Links%20Dev.csv
* If minor fixes are needed, create PRs and merge them as necessary if they are low risk. Ensure fixes are tested manually.
* If major fixes are needed, consider the severity and if they are release critical. If they are, consider delaying release to ensure the issue is fixed (and tested).

## Prior to release: 1 day out

* Ensure that the mainline code you are planning to release is stable: Benchmark, ensure CI passes, check with team, etc.
* Prepare the release notes located in `docs/whats_new/vX.Y.Z.md`:
  * This will be copy-pasted into GitHub when you release.
  * Include all merged PRs into the notes and mention all PR authors / contributors (refer to past releases for examples).
  * Prioritize major features before minor features when ordering, otherwise order by merge date.
  * Run the script `release_instructions/add_links_to_release_notes.py` to add links to all pull requests and GitHub users mentioned in the release notes.
  * Review with at least 2 core maintainers to ensure release notes are correct.
  * Merge a PR that adds the new `docs/whats_new/vX.Y.Z.md` file. Ensure you also update the `docs/whats_new/index.md` in the same PR.
    * DO NOT commit the `docs/whats_new/vX.Y.Z_paste_to_github.md` file that is created. This is only used for pasting the GitHub release notes.
* Cut a release branch with format `X.Y.Z` (no v) - this branch is required to publish docs to versioned path
  * Clone from master branch
  * Add 1 commit to the release branch to remove pre-release warnings and update install instructions to remove `--pre`: [Old diff](https://github.com/autogluon/autogluon/commit/1d66194d4685b06e884bbf15dcb97580cbfb9261)
  * Add 1 commit that converts notebook links from `master` to `stable` by running this command from the root project directory: `LC_ALL=C find docs/tutorials/ -type f -exec sed -i '' 's#blob/master/docs#blob/stable/docs#' {} +`
  * Update links to AG sub-modules website to be stable ones, i.e. cloud
  * Push release branch
  * Build the release branch docs in [CI](https://ci.gluon.ai/job/autogluon/).
  * Once CI passes, verify it's available at `https://auto.gluon.ai/0.x.y/index.html`

## Release

* Update the `stable` documentation to the new release:
  * Delete the `stable` branch.
  * Create new `stable` branch from `vX.Y.Z` branch (They should be identical).
  * Add and push any change in `docs/README.md` (i.e. space) to ensure `stable` branch is different from `0.x.y`. 
    * This is required for GH Action to execute CI continuous integration step if `vX.Y.Z` and `stable` hashes are matching.
  * Wait for CI build of the `stable` branch to pass
  * Check that website has updated to align with the release docs.
* Perform version release by going to https://github.com/autogluon/autogluon/releases and click 'Draft a new release' in top right.
  * Tag release with format `vX.Y.Z`
  * Name the release identically to the tag (ex: `vX.Y.Z`)
  * Select `master` branch as a target
    * Note: we generally use master unless there are certain commits there we don't want to add to the release
  * DO NOT use the 'Save draft' option during the creation of the release. This breaks GitHub pipelines.
  * Copy-paste the content of `docs/whats_new/vX.Y.Z_paste_to_github.md` into the release notes box.
    * If this file doesn't exist, run `release_instructions/add_links_to_release_notes.py` to generate it.
    * DO NOT use `docs/whats_new/vX.Y.Z.md` -> This will break GitHub's contributor detection logic due to the URLs present around the GitHub aliases. This is why we need to use the `_paste_to_github.md` variant.
    * Ensure release notes look correct and make any final formatting fixes.
  * Click 'Publish release' and the release will go live.
* Wait ~10 minutes and then locally test that the PyPi package is available and working with the latest release version, ask team members to also independently verify.

## Conda-Forge Release

After GitHub & PyPi release, conduct release on Conda-Forge
* Please refer to [conda release instructions](update-conda-recipes.md) for details.

## Release Cheatsheet

* If intending to create a new cheatsheet for the release, refer to [autogluon-doc-utils README.md](https://github.com/Innixma/autogluon-doc-utils) for instructions on creating a new cheatsheet.
* If a cheatsheet exists for `0.x.y` (or `0.x`), update the `docs/cheatsheet.md` url paths ([example](https://github.com/autogluon/autogluon/blob/0.4.1/docs/cheatsheet.rst)) in branch `0.x.y` to the correct location ([example for v0.4.0 and v0.4.1](https://github.com/Innixma/autogluon-doc-utils/tree/main/docs/cheatsheets/v0.4.0)).
  * Example urls: [JPEG](https://raw.githubusercontent.com/Innixma/autogluon-doc-utils/main/docs/cheatsheets/v0.4.0/autogluon-cheat-sheet.jpeg), [PDF](https://nbviewer.org/github/Innixma/autogluon-doc-utils/blob/main/docs/cheatsheets/v0.4.0/autogluon-cheat-sheet.pdf)
  * Do NOT do this for `stable` branch or `master` branch, instead have them continue pointing to the [stable cheatsheet files](https://github.com/Innixma/autogluon-doc-utils/tree/main/docs/cheatsheets/stable). This is to ensure that as we release new versions of the cheatsheet, old docs will still refer to the correct cheatsheet for their version.
  * Finally, update the stable files [here](https://github.com/Innixma/autogluon-doc-utils/tree/main/docs/cheatsheets/stable) to reflect the latest released version of the cheatsheet.

## Post Release

* IF THERE IS A MAJOR ISSUE: Do an emergency hot-fix and a new release ASAP. Releases cannot be deleted, so a new release will have to be done.

After release is published, on the mainline branch:
* Update `release` in `docs/conf.py`
* Increment version in the `VERSION` file and `SECURITY.md`
* Update doc links in `docs/versions.rst`
* Update `README.md` sample code with new release version.
* Send release update to internal and external slack channels and mailing lists
* Publish any blogs / talks planned for release to generate interest.

## Post Release Conda-Forge Patching

Conda-Forge releases are mutable and can be changed post-release to fix breaking bugs without releasing a new version.

* Create a new branch in your forked `autogluon.{module}-feedstock` repo
* Make necessary updates on packages for patching
* Increment the `number` field under `build` by 1 and keep the rest of `package` and `source` information unchanged
* Refer to [conda release instructions](update-conda-recipes.md) for more details
