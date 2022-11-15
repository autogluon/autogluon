echo "1"
python3 watercolor_benchmark.py \
        --checkpoint_name vfnet_r50_fpn_mdconv_c3-c5_mstrain_2x_coco \
        --lr 5e-4 \
        --per_gpu_batch_size 4 \
        > vfnet_r50_watercolor_lr5e-4.txt

echo "2"
python3 watercolor_benchmark.py \
        --checkpoint_name vfnet_r50_fpn_mdconv_c3-c5_mstrain_2x_coco \
        --lr 1e-3 \
        --per_gpu_batch_size 4 \
        > vfnet_r50_watercolor_lr1e-3.txt

echo "3"
python3 watercolor_benchmark.py \
        --checkpoint_name vfnet_r50_fpn_mdconv_c3-c5_mstrain_2x_coco \
        --lr 5e-3 \
        --per_gpu_batch_size 4 \
        > vfnet_r50_watercolor_lr5e-3.txt

echo "4"
python3 watercolor_benchmark.py \
        --checkpoint_name vfnet_x101_64x4d_fpn_mdconv_c3-c5_mstrain_2x_coco \
        --lr 1e-4 \
        --per_gpu_batch_size 2 \
        > vfnet_x101_watercolor_lr1e-4.txt

echo "4"
python3 watercolor_benchmark.py \
        --checkpoint_name vfnet_x101_64x4d_fpn_mdconv_c3-c5_mstrain_2x_coco \
        --lr 5e-4 \
        --per_gpu_batch_size 2 \
        > vfnet_x101_watercolor_lr5e-4.txt

echo "5"
python3 watercolor_benchmark.py \
        --checkpoint_name vfnet_x101_64x4d_fpn_mdconv_c3-c5_mstrain_2x_coco \
        --lr 1e-3 \
        --per_gpu_batch_size 2 \
        > vfnet_x101_watercolor_lr1e-3.txt
