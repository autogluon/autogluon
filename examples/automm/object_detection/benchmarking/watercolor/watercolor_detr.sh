echo "1"
python3 watercolor_benchmark.py \
        --checkpoint_name detr_r50_8x2_150e_coco \
        --lr 5e-4 \
        --per_gpu_batch_size 4 \
        > detr_r50_watercolor_lr5e-4.txt

echo "2"
python3 watercolor_benchmark.py \
        --checkpoint_name detr_r50_8x2_150e_coco \
        --lr 1e-3 \
        --per_gpu_batch_size 4 \
        > detr_r50_watercolor_lr1e-3.txt

echo "3"
python3 watercolor_benchmark.py \
        --checkpoint_name detr_r50_8x2_150e_coco \
        --lr 5e-3 \
        --per_gpu_batch_size 4 \
        > detr_r50_watercolor_lr5e-3.txt

echo "4"
python3 watercolor_benchmark.py \
        --checkpoint_name deformable_detr_twostage_refine_r50_16x2_50e_coco \
        --lr 5e-4 \
        --per_gpu_batch_size 4 \
        > dedetr_r50_watercolor_lr5e-4.txt

echo "5"
python3 watercolor_benchmark.py \
        --checkpoint_name deformable_detr_twostage_refine_r50_16x2_50e_coco \
        --lr 1e-3 \
        --per_gpu_batch_size 4 \
        > dedetr_r50_watercolor_lr1e-3.txt

echo "6"
python3 watercolor_benchmark.py \
        --checkpoint_name deformable_detr_twostage_refine_r50_16x2_50e_coco \
        --lr 5e-3 \
        --per_gpu_batch_size 4 \
        > dedetr_r50_watercolor_lr5e-3.txt
