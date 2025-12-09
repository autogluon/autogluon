function setup_mmcv {
  if [[ $(python3 -c "import sys; print(sys.version_info >= (3, 13))") == "True" ]]; then
    echo "Skipping MMCV installation on Python 3.13 (not supported)"
    return 0
  fi
  # Install MMEngine from git with the fix for torch 2.5
  python3 -m pip install "git+https://github.com/open-mmlab/mmengine.git@2e0ab7a92220d2f0c725798047773495d589c548"
  mim install "mmcv==2.1.0" --timeout 60
  python3 -m pip install "mmdet==3.2.0"
}
