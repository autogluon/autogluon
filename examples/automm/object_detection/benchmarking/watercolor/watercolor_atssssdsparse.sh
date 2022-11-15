echo "1"
python3 watercolor_benchmark.py \
        --checkpoint_name atss_r50_fpn_dyhead_1x_coco \
        --lr 5e-4 \
        --per_gpu_batch_size 4 \
        > atss_r50_watercolor_lr5e-4.txt

echo "2"
python3 watercolor_benchmark.py \
        --checkpoint_name atss_r50_fpn_dyhead_1x_coco \
        --lr 1e-3 \
        --per_gpu_batch_size 4 \
        > atss_r50_watercolor_lr1e-3.txt

echo "3"
python3 watercolor_benchmark.py \
        --checkpoint_name atss_r50_fpn_dyhead_1x_coco \
        --lr 5e-3 \
        --per_gpu_batch_size 4 \
        > atss_r50_watercolor_lr5e-3.txt

echo "4"
python3 watercolor_benchmark.py \
        --checkpoint_name sparse_rcnn_r50_fpn_300_proposals_crop_mstrain_480-800_3x_coco \
        --lr 5e-4 \
        --per_gpu_batch_size 4 \
        > sparse_r50_watercolor_lr5e-4.txt

echo "5"
python3 watercolor_benchmark.py \
        --checkpoint_name sparse_rcnn_r50_fpn_300_proposals_crop_mstrain_480-800_3x_coco \
        --lr 1e-3 \
        --per_gpu_batch_size 4 \
        > sparse_r50_watercolor_lr1e-3.txt

echo "6"
python3 watercolor_benchmark.py \
        --checkpoint_name sparse_rcnn_r50_fpn_300_proposals_crop_mstrain_480-800_3x_coco \
        --lr 5e-3 \
        --per_gpu_batch_size 4 \
        > sparse_r50_watercolor_lr5e-3.txt

echo "7"
python3 watercolor_benchmark.py \
        --checkpoint_name ssdlite_mobilenetv2_scratch_600e_coco \
        --lr 5e-4 \
        --per_gpu_batch_size 32 \
        > ssdlite_mv2_watercolor_lr5e-4.txt

echo "8"
python3 watercolor_benchmark.py \
        --checkpoint_name ssdlite_mobilenetv2_scratch_600e_coco \
        --lr 1e-3 \
        --per_gpu_batch_size 32 \
        > ssdlite_mv2_watercolor_lr1e-3.txt

echo "9"
python3 watercolor_benchmark.py \
        --checkpoint_name ssdlite_mobilenetv2_scratch_600e_coco \
        --lr 5e-3 \
        --per_gpu_batch_size 32 \
        > ssdlite_mv2_watercolor_lr5e-3.txt

