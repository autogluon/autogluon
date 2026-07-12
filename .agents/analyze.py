#!/usr/bin/env python3
"""
Generate machine-readable agent assets for AutoGluon by static analysis.

Outputs (written next to this script):
  - dependency_graph.json   inter-package `import autogluon.X` edges
  - packages.json           per-package stats + test counts
  - api_surface.json        public classes + top-level functions per package
  - entry_points.json       Predictor/Space/data classes with file:line

Methodology: AST extraction (no execution, no network). Re-run after
restructuring to keep the agent docs honest.

Usage:  python3 .agents/analyze.py   (from repo root)
"""
import ast
import json
import os
import re
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MEMBERS = ["common", "core", "features", "multimodal", "tabular", "timeseries"]
OUT = os.path.dirname(os.path.abspath(__file__))

ENTRY_RE = re.compile(r"(Predictor|Space|RFDataset|TimeSeriesDataFrame)")


def member_imports(tree):
    """Return the set of autogluon.<member> names this AST imports."""
    deps = set()
    for node in ast.walk(tree):
        mod = None
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.startswith("autogluon."):
                    mod = a.name
        elif isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("autogluon"):
            mod = node.module
        if mod:
            parts = mod.split(".")
            if len(parts) > 1 and parts[1] in MEMBERS:
                deps.add(parts[1])
    return deps


def _is_property(node):
    """True if a FunctionDef is a @property (or @x.setter) — not a real method."""
    return any(isinstance(d, ast.Name) and d.id == "property"
               or isinstance(d, ast.Attribute) and d.attr == "property"
               for d in node.decorator_list)


def public_defs(tree):
    classes, funcs = [], []
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
            methods = [n.name for n in node.body
                       if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                       and not n.name.startswith("_")
                       and not _is_property(n)]
            classes.append({"name": node.name, "methods": methods, "lineno": node.lineno})
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not node.name.startswith("_"):
            funcs.append({"name": node.name, "lineno": node.lineno})
    return classes, funcs


def count_tests(pkg):
    tdir = os.path.join(ROOT, pkg, "tests")
    n = 0
    if os.path.isdir(tdir):
        for _r, _d, files in os.walk(tdir):
            n += sum(1 for f in files if "test" in f and f.endswith(".py"))
    return n


def py_files(pkg):
    base = os.path.join(ROOT, pkg, "src", "autogluon", pkg)
    out = []
    if os.path.isdir(base):
        for r, _d, files in os.walk(base):
            for f in files:
                if f.endswith(".py"):
                    out.append(os.path.join(r, f))
    return base, out


# ---- collect ----
pkg_stats = {}
api_surface = {}
entry_points = []
edges = defaultdict(list)
self_edges = defaultdict(list)

for pkg in MEMBERS:
    base, files = py_files(pkg)
    n_cls = n_fn = 0
    pkg_classes = []
    pkg_funcs = []
    pkg_deps = set()
    for p in files:
        try:
            tree = ast.parse(open(p, encoding="utf-8").read())
        except Exception:
            continue
        cls, fns = public_defs(tree)
        n_cls += len(cls)
        n_fn += len(fns)
        pkg_classes += [{"file": os.path.relpath(p, ROOT), **c} for c in cls]
        pkg_funcs += [{"file": os.path.relpath(p, ROOT), **f} for f in fns]
        deps = member_imports(tree)
        pkg_deps |= deps
        for d in deps:
            (self_edges[pkg] if d == pkg else edges[pkg]).append(d) if False else None
            if d != pkg:
                edges[pkg].append(d)
        # entry points
        for c in cls:
            if ENTRY_RE.search(c["name"]):
                entry_points.append({
                    "package": pkg, "name": c["name"],
                    "file": os.path.relpath(p, ROOT), "line": c["lineno"],
                    "methods": c["methods"],
                })

    pkg_stats[pkg] = {
        "import_name": f"autogluon.{pkg}",
        "src_dir": os.path.relpath(base, ROOT),
        "py_files": len(files),
        "public_classes": n_cls,
        "public_functions": n_fn,
        "depends_on": sorted(pkg_deps - {pkg}),
        "test_files": count_tests(pkg),
    }
    api_surface[pkg] = {"classes": pkg_classes, "functions": pkg_funcs}

# dedupe edges
edges = {k: sorted(set(v)) for k, v in edges.items()}


def write(name, obj):
    with open(os.path.join(OUT, name), "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False)
        fh.write("\n")


write("dependency_graph.json", {
    "generated_by": ".agents/analyze.py (AST)",
    "packages": MEMBERS,
    "edges": {f"autogluon.{k}": [f"autogluon.{v}" for v in vs] for k, vs in edges.items()},
    "notes": "edges = at least one `import autogluon.<dst>` in src. self-imports omitted.",
})
write("packages.json", {"packages": pkg_stats})
write("api_surface.json", api_surface)
write("entry_points.json", {"entry_points": entry_points})

print("Wrote: dependency_graph.json, packages.json, api_surface.json, entry_points.json")
for pkg in MEMBERS:
    s = pkg_stats[pkg]
    print(f"  {pkg:11} files={s['py_files']:3} cls={s['public_classes']:3} "
          f"fn={s['public_functions']:3} tests={s['test_files']:3} deps={s['depends_on']}")
