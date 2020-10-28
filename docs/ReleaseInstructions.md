### Release process

* Update version `title` in `root_index.html`.
* Tag release with format `v0.0.x`
* Cut a release branch with format `0.0.x` (no v) - this branch is required to publish docs to versioned path
* Push release branch and tags
* Build the release branch docs in [CI](https://ci.gluon.ai/job/autogluon/).
* Verify it's available at `https://auto.gluon.ai/0.0.x/index.html`
* Ensure that the mainline code you are planning to release is stable: Benchmark, ensure CI passes, check with team, etc.
* Ensure latest pre-release is working via `pip install --pre --upgrade autogluon` and testing to get an idea of how the actual release will function (Ideally with fresh venv). DO NOT RELEASE if the pre-release does not work.
* Perform version release by going to https://github.com/awslabs/autogluon/releases and click 'Draft a new release' in top right.
* DO NOT use the 'Save draft' option during the creation of the release. This breaks GitHub pipelines.
* Name the release identically to the tag (ex: `v0.x.y`)
* Include all merged PRs into the notes and mention all PR authors / contributors (refer to past releases for examples). Prioritize major features before minor features when ordering, otherwise order by merge date.
* Click 'Publish release' and the release will go live.
* Wait ~10 minutes and then locally test that the PyPi package is available and working with the latest release version, ask team members to also independently verify.
* IF THERE IS A MAJOR ISSUE: Do an emergency hot-fix and a new release ASAP. Releases cannot be deleted, so a new release will have to be done.
* Send release update to internal and external slack channels and mailing lists
* Publish any blogs / talks planned for release to generate interest.

After release is published, on the mainline branch:
* Update header links pointing to the **previous** version in `docs/config.ini` 
    ```
    header_links = v0.0.x Documentation, https://auto.gluon.ai/0.0.x/index.html, fas fa-book,
    ```
* Update `release` in `docs/config.ini`
* Increment version in the `VERSION` file
* Add new version links to `docs/versions.rst`
