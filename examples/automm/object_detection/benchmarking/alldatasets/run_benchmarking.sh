#!/bin/bash

# TODO: some datasets need preprocessing
# Kitchen: filename in annotation is part of the actual filename

dataset_names=("KITTI" "KITTI")
short_checkpoint_names=("yolov3_mv2" "yolov3_d53")
lr_modes=("low" "med")
#dataset_names=("KITTI" "LISA" "clipart" "comic" "deeplesion" "dota" "watercolor" "widerface" "VOC" "pothole")
#short_checkpoint_names=("centernet_r18" "yolov3_mv2" "yolov3_d53" "cascadercnn_r50" "vfnet_r50" "vfnet_x101")
#short_checkpoint_names=("yolov3_mv2" "vfnet_r50" "vfnet_x101")
#lr_modes=("low" "med" "high")

for d in ${dataset_names[*]}
do
  for c in ${short_checkpoint_names[*]}
  do
    for l in ${lr_modes[*]}
    do
      python3 single_run.py \
        -d $d \
        -c $c \
        -l $l \
        >& logs/${d}_${c}_${l}_log.txt
    done
  done
done