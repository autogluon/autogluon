function setup_mmcv {
  mim install "mmcv==2.1.0" --timeout 600
  python3 -m pip install "mmdet>=3.0.0"
}
