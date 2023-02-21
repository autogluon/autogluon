:::{note} LightGBM support on MacOS (LibOMP)
AutoGluon dependency LightGBM uses `libomp` for multi-threading. If you install `libomp` via `brew install libomp`, you may get segmentation faults due to incompatible library versions. Install a compatible version using the following commands:

```bash
# Uninstall libomp if it was previous installed
brew uninstall -f libomp
wget https://raw.githubusercontent.com/Homebrew/homebrew-core/fb8323f2b170bd4ae97e1bac9bf3e2983af3fdb0/Formula/libomp.rb
brew install libomp.rb
```
:::

