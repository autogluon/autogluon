---
name: autogluon-conda-upgrade
description: Automate AutoGluon conda-forge feedstock version upgrades. Use when the user wants to upgrade AutoGluon to a new version in conda-forge, create PRs for AutoGluon conda feedstocks, or update autogluon.common, autogluon.core, autogluon.features, autogluon.tabular, autogluon.multimodal, autogluon.timeseries, or autogluon meta-package feedstocks.
---

# AutoGluon Conda Feedstock Upgrade Workflow

**CRITICAL: DO NOT MERGE PULL REQUESTS.** Only create PRs. The user will review and merge them manually.

## Step 1: Prerequisites Check

### 1.1 Check GitHub CLI

```bash
gh --version
```

If not installed, stop and tell the user to install from https://github.com/cli/cli#installation and run `gh auth login`.

### 1.2 Check Authentication

```bash
gh auth status
```

If not authenticated, ask user to run `gh auth login`.

### 1.3 Gather User Input

Ask for:
- **New AutoGluon version number** (e.g., `1.5.0`)
- **Working directory** (default: `~/autogluon-feedstock-upgrade`)

## Step 2: Setup Working Directory and Fork/Clone Repos

```bash
mkdir -p {WORKING_DIR}
cd {WORKING_DIR}

gh repo fork conda-forge/autogluon.common-feedstock --clone=true --remote=true
gh repo fork conda-forge/autogluon.features-feedstock --clone=true --remote=true
gh repo fork conda-forge/autogluon.core-feedstock --clone=true --remote=true
gh repo fork conda-forge/autogluon.tabular-feedstock --clone=true --remote=true
gh repo fork conda-forge/autogluon.multimodal-feedstock --clone=true --remote=true
gh repo fork conda-forge/autogluon.timeseries-feedstock --clone=true --remote=true
gh repo fork conda-forge/autogluon-feedstock --clone=true --remote=true
```

## Step 3: Compute SHA256 Hash

```bash
curl -sL "https://github.com/autogluon/autogluon/archive/refs/tags/v{NEW_VERSION}.tar.gz" -o /tmp/autogluon-{NEW_VERSION}.tar.gz
openssl sha256 /tmp/autogluon-{NEW_VERSION}.tar.gz | awk '{print $2}'
rm /tmp/autogluon-{NEW_VERSION}.tar.gz
```

If curl fails with 404, ask user to verify the version number.

## Step 4: Fetch and Analyze Dependencies

### 4.1 Get Current (Old) Version

Read from `{WORKING_DIR}/autogluon.common-feedstock/recipe/meta.yaml`:
```
{% set version = "X.Y.Z" %}
```

### 4.2 Fetch Version Bounds

Fetch `_setup_utils.py` for both versions:
- New: `https://raw.githubusercontent.com/autogluon/autogluon/refs/tags/v{NEW_VERSION}/core/src/autogluon/core/_setup_utils.py`
- Old: `https://raw.githubusercontent.com/autogluon/autogluon/refs/tags/v{OLD_VERSION}/core/src/autogluon/core/_setup_utils.py`

Extract `DEPENDENT_PACKAGES` dictionary and `PYTHON_REQUIRES` string.

### 4.3 Fetch Package-Specific Setup Files

For each subpackage (common, features, core, tabular, multimodal, timeseries, autogluon), fetch:
`https://raw.githubusercontent.com/autogluon/autogluon/refs/tags/v{NEW_VERSION}/{SUBPACKAGE}/setup.py`

The `install_requires` shows which `DEPENDENT_PACKAGES` each subpackage needs.

### 4.4 Create Dependency Change Summary

Compare old vs new. Summarize:
1. Changed version bounds
2. Added dependencies
3. Removed dependencies
4. Python version changes

**Present summary to user and ask for confirmation before proceeding.**

## Step 5: Update Each Feedstock

Process in **dependency order**:

| Order | Feedstock | Dependencies |
|-------|-----------|--------------|
| 1 | autogluon.common-feedstock | (none) |
| 2 | autogluon.features-feedstock | common |
| 3 | autogluon.core-feedstock | common |
| 4 | autogluon.tabular-feedstock | core, features |
| 5 | autogluon.multimodal-feedstock | core |
| 6 | autogluon.timeseries-feedstock | core, tabular |
| 7 | autogluon-feedstock | all subpackages |

### For Each Feedstock:

#### 5.1 Sync Fork and Create Branch (DO THIS FIRST)

```bash
cd {WORKING_DIR}/{FEEDSTOCK_NAME}
git fetch upstream
git checkout main
git reset --hard upstream/main
git checkout -b {NEW_VERSION}
```

#### 5.2 Read Current meta.yaml

After creating branch, read `recipe/meta.yaml` to understand current structure.

#### 5.3 Update meta.yaml

1. **Update version:** `{% set version = "{NEW_VERSION}" %}`
2. **Update sha256:** Use computed hash
3. **Reset build number:** `number: 0`
4. **Update Python version** (if changed): `python >={{ python_min }},<{NEW_PYTHON_MAX}`
5. **Update dependency version bounds:** Match `DEPENDENT_PACKAGES`

**Rules:**
- Keep `autogluon.*` dependencies as `=={{ version }}`
- Only include dependencies from that package's `setup.py`
- Preserve existing comments
- Use conda naming (see Package Name Mappings below)

#### 5.4 Handle python_min Changes

If minimum Python changed, update `.ci_support/linux_64_.yaml`:
```yaml
python_min:
- '{NEW_PYTHON_MIN}'
```

## Step 6: Commit and Push

For each feedstock:
```bash
cd {WORKING_DIR}/{FEEDSTOCK_NAME}
git add recipe/meta.yaml
git commit -m "Update to v{NEW_VERSION}"
git push -u origin {NEW_VERSION}
```

## Step 7: Create Pull Requests

For each feedstock:
```bash
cd {WORKING_DIR}/{FEEDSTOCK_NAME}
gh pr create \
  --repo conda-forge/{FEEDSTOCK_NAME} \
  --title "Update to v{NEW_VERSION}" \
  --body "$(cat <<'EOF'
## Summary
- Update {PACKAGE_NAME} to version {NEW_VERSION}
- Updated dependency version bounds from upstream

## Dependency Changes
{LIST_RELEVANT_CHANGES}

## Checklist
* [x] Used a personal fork of the feedstock to propose changes
* [x] Reset the build number to `0`
* [ ] Re-rendered (Use `@conda-forge-admin, please rerender` in a comment)
EOF
)"
```

## Step 8: Final Summary

### 8.1 Provide PR Links

List all 7 created PRs with clickable links.

### 8.2 Merge Order Reminder

> **Merge PRs in dependency order:**
> 1. `autogluon.common` (no dependencies)
> 2. `autogluon.features` and `autogluon.core` (parallel)
> 3. `autogluon.tabular` and `autogluon.multimodal` (parallel)
> 4. `autogluon.timeseries`
> 5. `autogluon` (meta-package)

### 8.3 Post-Merge Instructions

> After each PR's CI passes:
> 1. Comment: `@conda-forge-admin, please rerender`
> 2. Wait for rerender bot to update
> 3. Once CI passes again, merge
> 4. Wait for package to be published before merging dependent PRs

---

## Appendix A: Dependency Tree

```
autogluon.common (base - no AG deps)
    │
    ├── autogluon.features (depends: common)
    │
    ├── autogluon.core (depends: common)
    │       │
    │       ├── autogluon.tabular (depends: core, features)
    │       │
    │       ├── autogluon.multimodal (depends: core)
    │       │
    │       └── autogluon.timeseries (depends: core, tabular)
    │
    └── autogluon [meta-package] (depends: all subpackages)
```

## Appendix B: URL Patterns

| Resource | URL |
|----------|-----|
| Release tarball | `https://github.com/autogluon/autogluon/archive/refs/tags/v{VERSION}.tar.gz` |
| Version bounds file | `https://raw.githubusercontent.com/autogluon/autogluon/refs/tags/v{VERSION}/core/src/autogluon/core/_setup_utils.py` |
| Package setup.py | `https://raw.githubusercontent.com/autogluon/autogluon/refs/tags/v{VERSION}/{SUBPACKAGE}/setup.py` |

## Appendix C: Sample PRs

- [autogluon.common PR #6](https://github.com/conda-forge/autogluon.common-feedstock/pull/6/files)
- [autogluon.features PR #5](https://github.com/conda-forge/autogluon.features-feedstock/pull/5/files)
- [autogluon.core PR #8](https://github.com/conda-forge/autogluon.core-feedstock/pull/8/files)
- [autogluon.tabular PR #15](https://github.com/conda-forge/autogluon.tabular-feedstock/pull/15/files)
- [autogluon.multimodal PR #16](https://github.com/conda-forge/autogluon.multimodal-feedstock/pull/16/files)
- [autogluon.timeseries PR #7](https://github.com/conda-forge/autogluon.timeseries-feedstock/pull/7/files)
- [autogluon PR #6](https://github.com/conda-forge/autogluon-feedstock/pull/6/files)

## Appendix D: Package Name Mappings

| PyPI Name | Conda-Forge Name |
|-----------|------------------|
| torch | pytorch |
| Pillow | pillow |
| scikit-learn | scikit-learn |
| PyYAML | pyyaml |
| opencv-python | opencv |
| tensorflow | tensorflow |
