function setup_mmcv {
  mim install mmcv --timeout 60
  python3 -m pip install "mmdet>=3.0.0"
}
