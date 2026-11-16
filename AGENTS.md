# Agent instructions

Instructions for coding agents working in this repository.

## Tests

New functionality, and fixes for bugs that can recur, come with tests. Those tests are part of the
change: they stay in the pull request and run in CI.

A test belongs in the pull request when it guards behavior that later, unrelated changes could
plausibly break:

- new functionality or a public API contract (arguments, return values, saved artifacts);
- an interface shared by many models or modules (a base-class method, `FitHelper`);
- a class of bug, tested where it can recur, not at the one line that was fixed.

Do not add a test whose only purpose is to detect that a change is reverted: that a parameter is now
passed through, a default changed, a call happens, a value is seeded. Review and git history protect
such a change, and the test would add runtime to every CI run.

The unit tests already take hours of CI compute, most of it spent fitting models, so keep each test as
cheap as its purpose allows:

- Extend an existing test (an assertion, a parametrization) before writing a new one.
- Do not fit a model unless the behavior only exists during a fit. Prefer stubs and small objects;
  otherwise use the existing `FitHelper` toy configurations with the smallest data that works.
- Do not add tests that need hardware CI does not have, such as several GPUs: CI skips them.

### Evidence for a change that needs no test

Some changes need no test under the rules above, yet a test is still the clearest way to show that
they work. Verify such a change locally (a scratch script, an existing test, a benchmark) and report the
result in the pull request description.

To let reviewers rerun that evidence, commit the test on the branch and remove it in a following commit
before asking for review. In the description, link that commit and give the command that runs the
test and its result (fails on the base, passes with the change). Pull requests are squash-merged, so
the test never reaches `master`. This applies only to such verification tests: a test that meets the
rules above stays in the pull request.

When reviewing a pull request, look for such a commit and run its test. Suggest keeping the test only if
it meets the rules above.

## Artifacts saved by older versions

Do not add logic whose only purpose is to keep loading a predictor or model that an older version of
AutoGluon saved. AutoGluon does not support backward compatibility of saved artifacts: by default a
predictor refuses to load under a version other than the one that created it
(`require_version_match=True`). A fallback for an outdated artifact never runs in supported use, and it
stays in the code long after anyone could need it.

- Read a new attribute directly. Do not write `getattr(self, "_new_attribute", default)` for an attribute
  that `__init__` or `fit` always sets, to cover an object pickled before the attribute existed.
- Do not re-apply a setting to a stored object on every call because an object saved earlier may lack it.
- Do not add a test that removes a new attribute, or rebuilds an old on-disk format, to show that an old
  artifact still loads.

This rule is about saved artifacts only. It does not change how a public argument, method or default is
deprecated, or which versions of a dependency the code supports.

When reviewing a pull request, ask for such logic and its tests to be removed.
