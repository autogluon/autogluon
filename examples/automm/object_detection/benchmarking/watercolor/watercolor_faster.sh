#echo "1"
#python3 watercolor_benchmark.py \
#        --checkpoint_name faster_rcnn_r50_fpn_2x_coco \
#        --lr 5e-4 \
#        --per_gpu_batch_size 4 \
#        > faster_r50_watercolor_lr5e-4_bs64.txt

#echo "2"
#python3 watercolor_benchmark.py \
#        --checkpoint_name faster_rcnn_r50_fpn_2x_coco \
#        --lr 1e-3 \
#        --per_gpu_batch_size 4 \
#        > faster_r50_watercolor_lr1e-3_bs64.txt

#echo "3"
#python3 watercolor_benchmark.py \
#        --checkpoint_name faster_rcnn_r50_fpn_2x_coco \
#        --lr 5e-3 \
#        --per_gpu_batch_size 4 \
#        > faster_r50_watercolor_lr5e-3_bs64.txt

echo "extra"
python3 watercolor_benchmark.py \
        --checkpoint_name faster_rcnn_r50_fpn_2x_coco \
        --lr 1e-4 \
        --per_gpu_batch_size 4 \
        > faster_r50_watercolor_lr1e-4_bs4.txt

echo "4"
python3 watercolor_benchmark.py \
        --checkpoint_name faster_rcnn_x101_32x4d_fpn_2x_coco \
        --lr 1e-4 \
        --per_gpu_batch_size 2 \
        > faster_x101_watercolor_lr1e-4_bs2.txt

echo "5"
python3 watercolor_benchmark.py \
        --checkpoint_name faster_rcnn_x101_32x4d_fpn_2x_coco \
        --lr 5e-4 \
        --per_gpu_batch_size 2 \
        > faster_x101_watercolor_lr5e-4_bs2.txt

echo "6"
python3 watercolor_benchmark.py \
        --checkpoint_name faster_rcnn_x101_32x4d_fpn_2x_coco \
        --lr 1e-3 \
        --per_gpu_batch_size 2 \
        > faster_x101_watercolor_lr1e-3_bs2.txt
