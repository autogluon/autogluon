#echo "1"
#python3 watercolor_benchmark.py \
#        --checkpoint_name centernet_resnet18_dcnv2_140e_coco \
#        --lr 1e-3 \
#        --per_gpu_batch_size 32 \
#        > center_r18_watercolor_lr1e-3.txt

echo "2"
python3 watercolor_benchmark.py \
        --checkpoint_name centernet_resnet18_dcnv2_140e_coco \
        --lr 5e-3 \
        --per_gpu_batch_size 32 \
        > center_r18_watercolor_lr5e-3.txt

echo "3"
python3 watercolor_benchmark.py \
        --checkpoint_name centernet_resnet18_dcnv2_140e_coco \
        --lr 1e-2 \
        --per_gpu_batch_size 32 \
        > center_r18_watercolor_lr1e-2.txt

echo "4"
python3 watercolor_benchmark.py \
        --checkpoint_name cascade_rcnn_r50_fpn_20e_coco \
        --lr 5e-4 \
        --per_gpu_batch_size 4 \
        > cascade_r50_watercolor_lr5e-4.txt

echo "5"
python3 watercolor_benchmark.py \
        --checkpoint_name cascade_rcnn_r50_fpn_20e_coco \
        --lr 1e-3 \
        --per_gpu_batch_size 4 \
        > cascade_r50_watercolor_lr1e-3.txt

echo "6"
python3 watercolor_benchmark.py \
        --checkpoint_name cascade_rcnn_r50_fpn_20e_coco \
        --lr 5e-3 \
        --per_gpu_batch_size 4 \
        > cascade_r50_watercolor_lr5e-3.txt

echo "7"
python3 watercolor_benchmark.py \
        --checkpoint_name cascade_rcnn_s101_fpn_syncbn-backbone+head_mstrain-range_1x_coco \
        --lr 1e-4 \
        --per_gpu_batch_size 2 \
        > cascade_s101_watercolor_lr1e-4.txt

echo "8"
python3 watercolor_benchmark.py \
        --checkpoint_name cascade_rcnn_s101_fpn_syncbn-backbone+head_mstrain-range_1x_coco \
        --lr 5e-4 \
        --per_gpu_batch_size 2 \
        > cascade_s101_watercolor_lr5e-4.txt

echo "9"
python3 watercolor_benchmark.py \
        --checkpoint_name cascade_rcnn_s101_fpn_syncbn-backbone+head_mstrain-range_1x_coco \
        --lr 1e-3 \
        --per_gpu_batch_size 2 \
        > cascade_s101_watercolor_lr1e-3.txt


