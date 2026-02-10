function setup_mmcv {
  if [[ $(python3 -c "import sys; print(sys.version_info >= (3, 13))") == "True" ]]; then
    echo "Skipping MMCV installation on Python 3.13 (not supported)"
    return 0
  fi
  # Install MMEngine from PyPI wheel to avoid setuptools>=82 removing pkg_resources
  python3 -m pip install "setuptools<82"
  python3 -m pip install "mmengine==0.10.5"
  python3 -m pip install "mmcv==2.1.0" --no-build-isolation --timeout 60
  python3 -m pip install "mmdet==3.2.0"
}
