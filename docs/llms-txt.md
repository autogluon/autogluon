# `llms.txt` — agent-friendly documentation

> This page documents the machine-readable documentation output, not a
> user-facing tutorial. It is intended for AI agent tooling and maintainers.

## What this is

AutoGluon's documentation is built with [Sphinx](https://www.sphinx-doc.org/)
and rendered to HTML for human reading. As of this change, the same build also
emits a **markdown mirror** of the documentation following the
[llms.txt convention](https://llmstxt.org/), so that AI coding agents and LLMs
can consume the docs without scraping HTML (which wastes tokens on navigation,
scripts and styling, and obscures the actual content).

The markdown is produced by [`sphinx-llm`](https://github.com/NVIDIA/sphinx-llm)
(the `sphinx_llm.txt` extension), registered in [`conf.py`](./conf.py).

## Outputs

A normal `sphinx-build -b html . _build/html/` (see
[`build_doc.sh`](./build_doc.sh)) now also writes, under `_build/html/`:

| file | contents |
|------|----------|
| `llms.txt` | a markdown index of all documentation pages, grouped by section, with one-line descriptions — the entry point an agent reads first |
| `llms-full.txt` | every documentation page concatenated into a single markdown file, for long-context models that want the whole corpus in one shot |
| `<page>.html.md` | a per-page markdown rendering of each page |

These files are served alongside the HTML site, so `https://auto.gluon.ai/llms.txt`
and `https://auto.gluon.ai/llms-full.txt` become available once deployed.

## Why sphinx-llm (and not sphinx-llms-txt)

AutoGluon's API reference under [`api/`](./api/) is generated dynamically via
`autosummary` / `automodule` / `autoclass` directives (see e.g.
[`api/tabular.rst`](./api/tabular.rst)). The simpler `sphinx-llms-txt` extension
reads the raw `.rst` source files and therefore leaves those directives
**unexpanded** — an agent would see literal `.. autoclass:: TabularPredictor`
placeholders instead of the rendered API.

`sphinx-llm` instead runs the full Sphinx build pipeline (autodoc, autosummary,
napoleon, myst-nb) and renders the *resolved* content to markdown. This means
docstrings, parameter tables, and type information are fully expanded in the
agent-facing output.

## Configuration

In [`conf.py`](./conf.py):

```python
extensions = [
    ...,
    "sphinx_llm.txt",
]

llms_txt_description = "AutoGluon automates machine learning tasks..."
```

`sphinx-llm` is added to [`requirements_doc.txt`](./requirements_doc.txt). It
runs automatically as part of the HTML build — no separate build step is
required. The `llms_txt_enabled` option can be set to `False` to disable it.

### When `llms.txt` is generated

`sphinx-llm` runs a second `sphinx-build` subprocess in parallel with the
main HTML build. On CI runners with constrained CPU/memory this **races the
parent process** during the expensive `nbsphinx` / `myst-nb` notebook
execution phase (`tutorials/*.ipynb`). Per-tutorial CI jobs therefore
**skip** the markdown build by default — they set no flag.

To opt back in for a one-off build, set the `AUTOGLUON_BUILD_ALL_DOCS`
environment variable:

```bash
export AUTOGLUON_BUILD_ALL_DOCS=1
sphinx-build -b html . _build/html
```

The `build_all_docs.sh` workflow script sets this automatically (it is the
only job that produces the deployed llms.txt artifacts served from
`auto.gluon.ai/llms.txt`).

## How an agent uses it

1. Fetch `https://auto.gluon.ai/llms.txt` to get the curated index of doc pages.
2. Either follow links to individual `*.html.md` pages for targeted context, or
3. Fetch `llms-full.txt` for the complete corpus (suitable for models with a
   large context window).

See the root [`AGENTS.md`](../AGENTS.md) for the companion code-level agent
assets (dependency graph, API surface, AST chunks) under `.agents/`.
