# Agent instructions

Instructions for coding agents working in this repository.

## Tests

The unit tests take hours of CI compute, most of it spent fitting models, so every new test is a cost
paid by every future pull request.

Add a test only when it guards behavior that later, unrelated changes could plausibly break:

- new functionality or a public API contract (arguments, return values, saved artifacts);
- an interface shared by many models or modules (a base-class method, `FitHelper`);
- a class of bug, tested where it can recur, not at the one line that was fixed.

Do not add a test whose only purpose is to detect that a change is reverted: that a parameter is now
passed through, a default changed, a call happens, a value is seeded. Review and git history protect
the change; such a test only adds runtime to every CI run.

When a test is warranted:

- Extend an existing test (an assertion, a parametrization) before writing a new one.
- Do not fit a model unless the behavior only exists during a fit. Prefer stubs and small objects;
  otherwise use the existing `FitHelper` toy configurations with the smallest data that works.
- Do not add tests that need hardware CI does not have, such as several GPUs: CI skips them.

### Showing that a fix works

Verify a fix locally (a scratch script, an existing test, a benchmark) and report the result in the
pull request description.

When a test is the clearest evidence, commit it on the branch and remove it in a following commit
before asking for review, and link that commit in the description with its result (fails on the
base, passes with the fix). Pull requests are squash-merged, so the test never reaches `master`, while
reviewers can check out the commit and run it.

When reviewing a pull request, look for such a commit and run its test, and suggest keeping the test
only if it meets the criteria above.
