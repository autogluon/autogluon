#!/bin/bash

# TODO: some datasets need preprocessing
# Kitchen: filename in annotation is part of the actual filename

#dataset_names=("KITTI" "pothole" "LISA" "clipart" "comic" "deeplesion" "dota" "watercolor" "widerface" "VOC")
dataset_names=("pothole" "clipart" "comic" "deeplesion" "dota" "watercolor" "widerface")
lr_modes=("med" "high" "low")

c="yolox_l"

for l in ${lr_modes[*]}
do
  for d in ${dataset_names[*]}
  do
    python3 single_run.py \
      -d $d \
      -c $c \
      -l $l \
      >& yolox_logs/${d}_${c}_${l}_log.txt
  done
done