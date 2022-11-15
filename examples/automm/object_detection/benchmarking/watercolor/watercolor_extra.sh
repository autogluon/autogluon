echo "cascade r50 1"
python3 watercolor_benchmark.py \
        --checkpoint_name cascade_rcnn_r50_fpn_20e_coco \
        --lr 1e-4 \
        --per_gpu_batch_size 4 \
        > cascade_r50_watercolor_lr1e-4.txt

echo "cascade r50 2"
python3 watercolor_benchmark.py \
        --checkpoint_name cascade_rcnn_r50_fpn_20e_coco \
        --lr 5e-5 \
        --per_gpu_batch_size 4 \
        > cascade_r50_watercolor_lr5e-5.txt

echo "cascade s101 1"
python3 watercolor_benchmark.py \
        --checkpoint_name cascade_rcnn_s101_fpn_syncbn-backbone+head_mstrain-range_1x_coco \
        --lr 1e-5 \
        --per_gpu_batch_size 2 \
        > cascade_s101_watercolor_lr1e-5.txt

echo "cascade s101 2"
python3 watercolor_benchmark.py \
        --checkpoint_name cascade_rcnn_s101_fpn_syncbn-backbone+head_mstrain-range_1x_coco \
        --lr 5e-5 \
        --per_gpu_batch_size 2 \
        > cascade_s101_watercolor_lr5e-5.txt

echo "detr 1"
python3 watercolor_benchmark.py \
        --checkpoint_name detr_r50_8x2_150e_coco \
        --lr 1e-2 \
        --per_gpu_batch_size 4 \
        > detr_r50_watercolor_lr1e-2.txt

echo "detr 2"
python3 watercolor_benchmark.py \
        --checkpoint_name detr_r50_8x2_150e_coco \
        --lr 5e-2 \
        --per_gpu_batch_size 4 \
        > detr_r50_watercolor_lr5e-2.txt

echo "dedetr 1"
python3 watercolor_benchmark.py \
        --checkpoint_name deformable_detr_twostage_refine_r50_16x2_50e_coco \
        --lr 5e-4 \
        --per_gpu_batch_size 2 \
        > dedetr_r50_watercolor_lr5e-4.txt

echo "dedetr 2"
python3 watercolor_benchmark.py \
        --checkpoint_name deformable_detr_twostage_refine_r50_16x2_50e_coco \
        --lr 1e-3 \
        --per_gpu_batch_size 2 \
        > dedetr_r50_watercolor_lr1e-3.txt

echo "dedetr 3"
python3 watercolor_benchmark.py \
        --checkpoint_name deformable_detr_twostage_refine_r50_16x2_50e_coco \
        --lr 5e-3 \
        --per_gpu_batch_size 2 \
        > dedetr_r50_watercolor_lr5e-3.txt

echo "vfnet r50 1"
python3 watercolor_benchmark.py \
        --checkpoint_name vfnet_r50_fpn_mdconv_c3-c5_mstrain_2x_coco \
        --lr 5e-5 \
        --per_gpu_batch_size 4 \
        > vfnet_r50_watercolor_lr5e-5.txt

echo "vfnet r50 2"
python3 watercolor_benchmark.py \
        --checkpoint_name vfnet_r50_fpn_mdconv_c3-c5_mstrain_2x_coco \
        --lr 1e-4 \
        --per_gpu_batch_size 4 \
        > vfnet_r50_watercolor_lr1e-4.txt

echo "vfnet x101 1"
python3 watercolor_benchmark.py \
        --checkpoint_name vfnet_x101_64x4d_fpn_mdconv_c3-c5_mstrain_2x_coco \
        --lr 5e-3 \
        --per_gpu_batch_size 2 \
        > vfnet_x101_watercolor_lr5e-3.txt

echo "vfnet x101 2"
python3 watercolor_benchmark.py \
        --checkpoint_name vfnet_x101_64x4d_fpn_mdconv_c3-c5_mstrain_2x_coco \
        --lr 1e-2 \
        --per_gpu_batch_size 2 \
        > vfnet_x101_watercolor_lr1e-2.txt

echo "atss 1"
python3 watercolor_benchmark.py \
        --checkpoint_name atss_r50_fpn_dyhead_1x_coco \
        --lr 5e-5 \
        --per_gpu_batch_size 4 \
        > atss_r50_watercolor_lr5e-5.txt

echo "atss 2"
python3 watercolor_benchmark.py \
        --checkpoint_name atss_r50_fpn_dyhead_1x_coco \
        --lr 1e-4 \
        --per_gpu_batch_size 4 \
        > atss_r50_watercolor_lr1e-4.txt

echo "sparse 1"
python3 watercolor_benchmark.py \
        --checkpoint_name sparse_rcnn_r50_fpn_300_proposals_crop_mstrain_480-800_3x_coco \
        --lr 5e-5 \
        --per_gpu_batch_size 4 \
        > sparse_r50_watercolor_lr5e-5.txt

echo "sparse 2"
python3 watercolor_benchmark.py \
        --checkpoint_name sparse_rcnn_r50_fpn_300_proposals_crop_mstrain_480-800_3x_coco \
        --lr 1e-4 \
        --per_gpu_batch_size 4 \
        > sparse_r50_watercolor_lr1e-4.txt

echo "ssdl 1"
python3 watercolor_benchmark.py \
        --checkpoint_name ssdlite_mobilenetv2_scratch_600e_coco \
        --lr 1e-2 \
        --per_gpu_batch_size 32 \
        > ssdlite_mv2_watercolor_lr1e-2.txt

echo "ssdl 2"
python3 watercolor_benchmark.py \
        --checkpoint_name ssdlite_mobilenetv2_scratch_600e_coco \
        --lr 5e-2 \
        --per_gpu_batch_size 32 \
        > ssdlite_mv2_watercolor_lr5e-2.txt
