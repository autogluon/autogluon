python3 watercolor_benchmark.py \
        --checkpoint_name yolov3_mobilenetv2_320_300e_coco \
        --lr 1e-3 \
        --per_gpu_batch_size 64 \
        > yolov3_mv2_watercolor_lr1e-3_bs64.txt

python3 watercolor_benchmark.py \
        --checkpoint_name yolov3_mobilenetv2_320_300e_coco \
        --lr 5e-3 \
        --per_gpu_batch_size 64 \
        > yolov3_mv2_watercolor_lr5e-3_bs64.txt

python3 watercolor_benchmark.py \
        --checkpoint_name yolov3_mobilenetv2_320_300e_coco \
        --lr 1e-2 \
        --per_gpu_batch_size 64 \
        > yolov3_mv2_watercolor_lr1e-2_bs64.txt

python3 watercolor_benchmark.py \
        --checkpoint_name yolov3_d53_mstrain-416_273e_coco \
        --lr 1e-3 \
        --per_gpu_batch_size 32 \
        > yolov3_d53_watercolor_lr1e-3_bs32.txt

python3 watercolor_benchmark.py \
        --checkpoint_name yolov3_d53_mstrain-416_273e_coco \
        --lr 5e-3 \
        --per_gpu_batch_size 32 \
        > yolov3_d53_watercolor_lr5e-3_bs32.txt

python3 watercolor_benchmark.py \
        --checkpoint_name yolov3_d53_mstrain-416_273e_coco \
        --lr 1e-2 \
        --per_gpu_batch_size 32 \
        > yolov3_d53_watercolor_lr1e-2_bs32.txt