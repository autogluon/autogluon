:::{admonition} LightGBM support on MacOS (LibOMP)
AutoGluon dependency LightGBM uses `libomp` for multi-threading. Install it via Homebrew:

```bash
brew install libomp
```

**Dual OpenMP (torch + LightGBM)**

On macOS, pip wheels of PyTorch (especially `torch>=2.11`), LightGBM, and scikit-learn can each resolve
a different `libomp.dylib`. Loading more than one OpenMP runtime in a process can SIGSEGV under
multi-threaded training
([AutoGluon#5793](https://github.com/autogluon/autogluon/issues/5793),
[pytorch#191933](https://github.com/pytorch/pytorch/issues/191933)).

AutoGluon aligns load paths automatically on import (macOS only) via
`autogluon.common.utils.macos_openmp`:

* **lightgbm**: relative rpath `@loader_path/../../torch/lib` (keeps `@rpath/libomp.dylib`)
* **scikit-learn**: symlink `sklearn/.dylibs/libomp.dylib` → torch’s vendored libomp

Requires a writable environment. Disable with:

```bash
export AUTOGLUON_DISABLE_MACOS_OPENMP_AUTOFIX=1
```

Optional CLI:

```bash
python -m autogluon.common.utils.macos_openmp fix
python -m autogluon.common.utils.macos_openmp check
python -m autogluon.common.utils.macos_openmp smoke
```
:::
