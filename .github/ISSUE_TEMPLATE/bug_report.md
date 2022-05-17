---
name: Bug Report
about: Create a report to help us reproduce and correct the bug
title: "[BUG]"
labels: ['bug: unconfirmed', 'Needs Triage']
assignees: ''

---

- [ ] I have checked that this bug exists on the latest stable version of AutoGluon
- [ ] and/or I have checked that this bug exists on the latest mainline of AutoGluon via source installation

**Describe the bug**
A clear and concise description of what the bug is.

**Expected behavior**
A clear and concise description of what you expected to happen.

**To Reproduce**
A minimal script to reproduce the issue. Links to Colab notebooks or similar tools are encouraged.  
If the code is too long, feel free to put it in a public gist and link it in the issue: https://gist.github.com.  
In short, we are going to copy-paste your code to run it and we expect to get the same result as you.

**Screenshots**
If applicable, add screenshots to help explain your problem.

**Installed Versions**
Which version of AutoGluon are you are using?  
If you are using 0.4.0 and newer, please run the following code snippet:
<details>

```python
# Replace this code with the output of the following:
from autogluon.core.utils import show_versions
show_versions()
```

</details>

**Additional context**
Add any other context about the problem here.
