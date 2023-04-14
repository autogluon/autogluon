---
name: Bug Report
about: Create a report to help us reproduce and correct the bug
title: "[BUG]"
labels: ['bug: unconfirmed', 'Needs Triage']
assignees: ''

---

**Bug Report Checklist**
<!-- Please ensure at least one of the following to help the developers troubleshoot the problem:  -->
- [ ] I provided code that demonstrates a minimal reproducible example. <!-- Ideal, especially via source install --> 
- [ ] I confirmed bug exists on the latest mainline of AutoGluon via source install. <!-- Preferred --> 
- [ ] I confirmed bug exists on the latest stable version of AutoGluon. <!-- Unnecessary if prior items are checked --> 

**Describe the bug**
<!-- A clear and concise description of what the bug is. -->

**Expected behavior**
<!-- A clear and concise description of what you expected to happen. -->

**To Reproduce**
<!-- A minimal script to reproduce the issue. Links to Colab notebooks or similar tools are encouraged.  
If the code is too long, feel free to put it in a public gist and link it in the issue: https://gist.github.com.  
In short, we are going to copy-paste your code to run it and we expect to get the same result as you. -->

**Screenshots / Logs**
<!-- If applicable, add screenshots or logs to help explain your problem. -->

**Installed Versions**
<!-- Please run the following code snippet: -->
<details>

```python
# Replace this code with the output of the following:
from autogluon.core.utils import show_versions
show_versions()
```

</details>
