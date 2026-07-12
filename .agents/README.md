# `.agents/` — machine-readable repo intelligence for AI agents

This directory holds **static-analysis artifacts** that give coding agents a
verifiable, up-to-date map of the AutoGluon monorepo. Unlike prose docs, every
number here is derived from the source by AST analysis and can be regenerated
deterministically.

## Contents

| file | what it is |
|------|-----------|
| `dependency_graph.json` | inter-package `import autogluon.X` edges |
| `packages.json`         | per-package stats (file/class/function counts, deps, test counts) |
| `api_surface.json`      | public classes + top-level functions per package, with `file:line` |
| `entry_points.json`     | Predictor/Space/data classes with `file:line` and method lists |
| `code_chunks.json`      | structure-aware (cAST-style) code chunks — **generated on demand** (gitignored, ~3MB) |
| `analyze.py`            | generator for all the JSON files above |

## Why this exists (vs. hand-written docs)

AutoGluon is a 7-member `uv` workspace whose structure is invisible from the
root. Prose READMEs describe *what the project does*; these artifacts describe
*how it is wired* — which package imports which, where each public class lives,
how big each surface is. An agent consulting `entry_points.json` can jump
straight to `TabularPredictor` at `tabular/.../predictor.py:91` without
grepping.

## Structure-aware code chunks (`code_chunks.json`)

This artifact applies **cAST-style chunking** (cAST, arXiv:2506.15655, EMNLP
2025 Findings): instead of splitting source by fixed character windows, code is
broken at AST boundaries — each class and function is one chunk, with its
method-level scope preserved. Every chunk carries a `context` field with a
self-locating header (file path, package, entity name, scope chain, member
signatures, line range) plus the file's imports, so a chunk retrieved into an
agent's context window is immediately self-describing.

**This file is gitignored** (~3MB; too volatile to commit). Generate it on demand:

```bash
python3 .agents/analyze.py   # produces code_chunks.json alongside the others
```

## Regenerating after structural changes

```bash
# from repo root:
python3 .agents/analyze.py          # refresh the JSON assets
# commit the small index files (dependency_graph/packages/api_surface/entry_points);
# code_chunks.json stays gitignored.
```

All statistics come straight from the analyzer — commit regenerated files
alongside the structural change that prompted them.

## Provenance

Approach informed by RepoAgent (OpenBMB, EMNLP 2024) — repository-level
documentation driven by AST dependency analysis — and Anthropic's
context-engineering guidance on progressive disclosure. The distinction here:
numbers and edges are machine-verified; only the human-facing role blurbs are
curated.
