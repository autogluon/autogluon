function setup_mmcv {
  mim install mmcv-full --timeout 60
  python3 -m pip install "mmdet>=2.28, <3.0.0"
}
