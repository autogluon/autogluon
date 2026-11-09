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
