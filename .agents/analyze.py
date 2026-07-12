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

# ---- cAST-style function-level chunking ----
# Structure-aware chunking: instead of fixed-size character splits, break code at
# AST entity boundaries (classes, methods, top-level functions) and prepend a
# `contextualizedText` header to each chunk carrying path, scope chain, imports,
# and neighbor signatures — so a retrieved chunk is self-locating for an agent.
# Methodology follows cAST (arXiv:2506.15655, EMNLP 2025 Findings). Implemented
# with the stdlib `ast` module (zero extra dependencies) rather than tree-sitter.

def _node_end(tree, node):
    """Best-effort end line for a node (stdlib ast lacks end_lineno on older Pythons)."""
    end = getattr(node, "end_lineno", None)
    if end:
        return end
    # fallback: scan descendants for the max lineno
    return max((getattr(n, "lineno", node.lineno) for n in ast.walk(node)), default=node.lineno)


def _signature(node):
    """One-line signature for a class/function, for neighbor context."""
    if isinstance(node, ast.ClassDef):
        return f"class {node.name}"
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        args = [a.arg for a in node.args.args if a.arg != "self"]
        kind = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
        return f"{kind} {node.name}({', '.join(args)})"
    return ""


def _scope_chain(parent_stack):
    """e.g. 'autogluon.tabular.TabularPredictor' — from outermost to immediate parent."""
    return ".".join(n.name for n in parent_stack if isinstance(n, (ast.ClassDef,)))


def chunk_file(path, tree, rel_path, pkg):
    """Yield cAST chunks for one file. Each chunk is a self-locating code unit."""
    # module-level imports (carried into every chunk's context header)
    imports = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            imports.append(ast.get_source_segment(open(path, encoding="utf-8").read(), node) or "")
    import_block = "\n".join(imports)[:1200]  # cap to keep chunks bounded

    src = open(path, encoding="utf-8").read()
    src_lines = src.splitlines()
    chunks = []
    parent_stack = []

    def walk(node, scope_parents):
        # top-level classes & functions, recursing into class bodies
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                # chunk the class body as one unit (cAST keeps the whole class together
                # when it fits; we don't split methods individually to preserve scope)
                start, end = child.lineno, _node_end(tree, child)
                cls_methods = [_signature(m) for m in child.body
                               if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))
                               and not m.name.startswith("_")]
                chunks.append(_make_chunk(rel_path, pkg, child.name, start, end,
                                          src_lines, import_block,
                                          scope=".".join(p.name for p in scope_parents + [child]),
                                          members=cls_methods,
                                          kind="class"))
                # recurse into the class to emit its methods as finer chunks
                walk(child, scope_parents + [child])
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and not child.name.startswith("_"):
                start, end = child.lineno, _node_end(tree, child)
                chunks.append(_make_chunk(rel_path, pkg, child.name, start, end,
                                          src_lines, import_block,
                                          scope=".".join(p.name for p in scope_parents),
                                          kind="method" if scope_parents else "function"))

    walk(tree, [])
    return chunks


def _make_chunk(rel_path, pkg, name, start, end, src_lines, import_block, scope, kind, members=None):
    # contextualizedText: a self-locating header (path, scope, imports, members).
    # NOTE: source body is intentionally NOT inlined — it already lives in the repo
    # at `file`:`start_line`-`end_line`. Keeping it out keeps this asset small and
    # its diff stable. An agent (or a RAG indexer) reads the body on demand.
    header_lines = [
        f"# path: {rel_path}",
        f"# package: autogluon.{pkg}",
        f"# entity: {kind} {name}",
    ]
    if scope:
        header_lines.append(f"# scope: {scope}")
    if members:
        header_lines.append(f"# members: {', '.join(members[:8])}")
    header_lines.append(f"# lines: {start}-{end}")
    return {
        "file": rel_path,
        "package": pkg,
        "entity": name,
        "kind": kind,
        "scope": scope,
        "start_line": start,
        "end_line": end,
        "loc": end - start + 1,
        "members": members[:12] if members else [],
        # compact context header + truncated import list (for retrieval/grounding)
        "context": "\n".join(header_lines) + "\n\n# imports:\n" + import_block[:600],
    }


all_chunks = []
for pkg in MEMBERS:
    base, files = py_files(pkg)
    for p in files:
        try:
            tree = ast.parse(open(p, encoding="utf-8").read())
        except Exception:
            continue
        rel = os.path.relpath(p, ROOT)
        all_chunks += chunk_file(p, tree, rel, pkg)

write("code_chunks.json", {
    "generated_by": ".agents/analyze.py (AST, cAST-style)",
    "methodology": "Structure-aware chunking at class/function boundaries with a "
                   "contextualizedText header (path/scope/imports). Follows cAST "
                   "(arXiv:2506.15655, EMNLP 2025 Findings). Stdlib ast, zero deps.",
    "total_chunks": len(all_chunks),
    "chunks": all_chunks,
})

print("Wrote: dependency_graph.json, packages.json, api_surface.json, "
      "entry_points.json, code_chunks.json")
for pkg in MEMBERS:
    s = pkg_stats[pkg]
    print(f"  {pkg:11} files={s['py_files']:3} cls={s['public_classes']:3} "
          f"fn={s['public_functions']:3} tests={s['test_files']:3} deps={s['depends_on']}")
