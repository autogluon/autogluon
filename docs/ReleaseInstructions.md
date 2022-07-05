# Release process

### Prior to release: 1 week out

* Ensure the version specified in `docs/config.ini`, `VERSION`, `docs/index.rst`, and `docs/badges.rst` align with the intended release version.
* Make final call for which in-progress PRs are release critical.
* Communicate with in-progress PR owners that code freeze is in effect, no PRs will be merged that are not release critical.
* Wait 1 day after code-freeze for pre-release to be published.
* Ensure latest pre-release is working via `pip install --pre autogluon` and testing to get an idea of how the actual release will function (Ideally with fresh venv). DO NOT RELEASE if the pre-release does not work.
* Ensure pip install instructions are working correctly for both CPU and GPU.
  * Ensure explicit torch installs have the correct version range and are not overwritten during `pip install autogluon`:
    * install-cpu-pip.rst
    * install-cpu-source.rst
    * install-windows-gpu.rst
* Ensure each sub-module is working IN ISOLATION via `pip install --pre autogluon.{submodule}`.
  * Ensure a fresh venv is used for each submodule.
  * Doing this will avoid issues like in v0.4 release with `autogluon.text` crashing when installed standalone due to missing setup.py dependencies
    * https://github.com/awslabs/autogluon/issues/1607
* If minor fixes are needed, create PRs and merge them as necessary if they are low risk. Ensure fixes are tested manually.
* If major fixes are needed, consider the severity and if they are release critical. If they are, consider delaying release to ensure the issue is fixed (and tested).

### Prior to release: 1 day out

* Ensure that the mainline code you are planning to release is stable: Benchmark, ensure CI passes, check with team, etc.
* Cut a release branch with format `0.x.y` (no v) - this branch is required to publish docs to versioned path
  * Clone from master branch
  * Add 1 commit to the release branch to remove pre-release warnings and update install instructions to remove `--pre`: https://github.com/awslabs/autogluon/commit/1d66194d4685b06e884bbf15dcb97580cbfb9261
  * Push release branch
  * Build the release branch docs in [CI](https://ci.gluon.ai/job/autogluon/).
  * Once CI passes, verify it's available at `https://auto.gluon.ai/0.x.y/index.html`
* Prepare the release notes located in `docs/whats_new/v0.x.y.md`:
  * This will be copy-pasted into GitHub when you release.
  * Include all merged PRs into the notes and mention all PR authors / contributors (refer to past releases for examples).
  * Prioritize major features before minor features when ordering, otherwise order by merge date.
  * Review with at least 2 core maintainers to ensure release notes are correct.

### Release

* Update the `stable` documentation to the new release:
  * Delete the `stable` branch.
  * Create new `stable` branch from `0.x.y` branch (They should be identical).
  * Wait for CI build of the `stable` branch to pass
  * Check that website has updated to align with the release docs.
* Perform version release by going to https://github.com/awslabs/autogluon/releases and click 'Draft a new release' in top right.
  * Tag release with format `v0.x.y`
  * Name the release identically to the tag (ex: `v0.x.y`)
  * DO NOT use the 'Save draft' option during the creation of the release. This breaks GitHub pipelines.
  * Copy-paste the content of `docs/whats_new/v0.x.y.md` into the release notes box.
    * Ensure release notes look correct and make any final formatting fixes.
  * Click 'Publish release' and the release will go live.
* Wait ~10 minutes and then locally test that the PyPi package is available and working with the latest release version, ask team members to also independently verify.

### Release Cheatsheet

* If intending to create a new cheatsheet for the release, refer to [autogluon-doc-utils README.md](https://github.com/Innixma/autogluon-doc-utils) for instructions on creating a new cheatsheet.
* If a cheatsheet exists for `0.x.y` (or `0.x`), update the `docs/cheatsheet.rst` url paths ([example](https://github.com/awslabs/autogluon/blob/0.4.1/docs/cheatsheet.rst)) in branch `0.x.y` to the correct location ([example for v0.4.0 and v0.4.1](https://github.com/Innixma/autogluon-doc-utils/tree/main/docs/cheatsheets/v0.4.0)).
  * Example urls: [JPEG](https://raw.githubusercontent.com/Innixma/autogluon-doc-utils/main/docs/cheatsheets/v0.4.0/autogluon-cheat-sheet.jpeg), [PDF](https://nbviewer.org/github/Innixma/autogluon-doc-utils/blob/main/docs/cheatsheets/v0.4.0/autogluon-cheat-sheet.pdf)
  * Do NOT do this for `stable` branch or `master` branch, instead have them continue pointing to the [stable cheatsheet files](https://github.com/Innixma/autogluon-doc-utils/tree/main/docs/cheatsheets/stable). This is to ensure that as we release new versions of the cheatsheet, old docs will still refer to the correct cheatsheet for their version.
  * Finally, update the stable files [here](https://github.com/Innixma/autogluon-doc-utils/tree/main/docs/cheatsheets/stable) to reflect the latest released version of the cheatsheet.

### Post Release

* IF THERE IS A MAJOR ISSUE: Do an emergency hot-fix and a new release ASAP. Releases cannot be deleted, so a new release will have to be done.

After release is published, on the mainline branch:
* Update header links pointing to the **previous** version in `docs/config.ini` 
    ```
    header_links = v0.x.y Documentation, https://auto.gluon.ai/0.x.y/index.html, fas fa-book,
    ```
* Update `release` in `docs/config.ini`
* Increment version in the `VERSION` file
* Update `ReleaseVersion` image link in `docs/badges.rst`
* Update `README.md` sample code with new release version.
* Add new version links to `docs/versions.rst`

* Send release update to internal and external slack channels and mailing lists
* Publish any blogs / talks planned for release to generate interest.
