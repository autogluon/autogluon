#!/bin/bash

# mmdet openimage models has denorm_bbox=True and cannot be trivially finetuned
# Kitchen: filename in annotation is part of the actual filename

#dataset_names=("KITTI" "pothole" "LISA" "clipart" "comic" "deeplesion" "dota" "watercolor" "widerface" "VOC")
dataset_names=("pothole" "clipart" "comic" "deeplesion" "dota" "watercolor" "widerface" "KITTI" "LISA")
lr_modes=("med" "high" "low")
short_checkpoint_names=("faster_r50_openimages" "retina_r50_openimages")

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
        >& openimages/${d}_${c}_${l}_log.txt
    done
  done
done