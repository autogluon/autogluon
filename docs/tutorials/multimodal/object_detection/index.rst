Object Detection
================

Pre-requisite
-------------
All detection modules depend on ``mmcv-full`` and ``mmdet`` packages.


To install ``mmcv-full``, run:

    ``mim install mmcv-full``

To install ``mmdet``, run:

    ``pip install mmdet``

For additional support, please refer to official instructions for mmdet_ and mmcv-full_

.. _mmdet: https://mmdetection.readthedocs.io/en/v2.2.1/install.html
.. _mmcv-full: https://mmcv.readthedocs.io/en/latest/get_started/installation.html


Quick Start
------------------
.. container:: cards

   .. card::
      :title: Quick Start on a Tiny COCO Format Dataset
      :link: quick_start/quick_start_coco.html

      How to train high quality object detection model with MultiModalPredictor in under 5 minutes on COCO format dataset.


Data Preparation
------------------
.. container:: cards

   .. card::
      :title: Prepare COCO2017 Dataset
      :link: data_preparation/prepare_coco17.html

      How to prepare COCO2017 dataset for object detection.

   .. card::
      :title: Prepare Pascal VOC Dataset
      :link: data_preparation/prepare_voc.html

      How to prepare Pascal VOC dataset for object detection.

   .. card::
      :title: Prepare Watercolor Dataset
      :link: data_preparation/prepare_watercolor.html

      How to prepare Watercolor dataset for object detection.

   .. card::
      :title: Convert VOC Format Dataset to COCO Format
      :link: data_preparation/voc_to_coco.html

      How to convert a dataset from VOC format to COCO format for object detection.


Finetune
------------------
.. container:: cards

   .. card::
      :title: Fast Finetune on COCO Format Dataset
      :link: finetune/detection_fast_finetune_coco.html

      How to fast finetune a pretrained model on a dataset in COCO format.

   .. card::
      :title: High Performance Finetune on COCO Format Dataset
      :link: finetune/detection_high_performance_finetune_coco.html

      How to finetune a pretrained model on a dataset in COCO format with high performance.


Inference
------------------
.. container:: cards

   .. card::
      :title: Inference using a pretrained model - COCO dataset
      :link: inference/detection_inference_coco.html

      How to inference with a pretrained model on COCO dataset

   .. card::
      :title: Inference using a pretrained model - VOC dataset
      :link: inference/detection_inference_voc.html

      How to inference with a pretrained model on VOC dataset


Evaluation
------------------
.. container:: cards

   .. card::
      :title: Evaluate Pretrained YOLOv3 on COCO Format Dataset
      :link: evaluation/detection_eval_yolov3_coco.html

      How to evaluate the very fast pretrained YOLOv3 model on dataset in COCO format

   .. card::
      :title: Evaluate Pretrained Faster R-CNN on COCO Format Dataset
      :link: evaluation/detection_eval_fasterrcnn_coco.html

      How to evaluate the pretrained Faster R-CNN model with high performance on dataset in COCO format

   .. card::
      :title: Evaluate Pretrained Deformable DETR on COCO Format Dataset
      :link: evaluation/detection_eval_ddetr_coco.html

      How to evaluate the pretrained Deformable DETR model with higher performance on dataset in COCO format

   .. card::
      :title: Evaluate Pretrained Faster R-CNN on VOC Format Dataset
      :link: evaluation/detection_eval_fasterrcnn_voc.html

      How to evaluate the pretrained Faster R-CNN model on dataset in VOC format


.. toctree::
   :maxdepth: 1
   :hidden:

   quick_start/index
   data_preparation/index
   finetune/index
   inference/index
   evaluation/index