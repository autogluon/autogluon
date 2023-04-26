# Object Detection

## Quick Start

::::{grid} 2
  :gutter: 3

:::{grid-item-card} Quick Start on a Tiny COCO Format Dataset
  :link: quick_start/quick_start_coco.html

  How to train high quality object detection model with MultiModalPredictor in under 5 minutes on COCO format dataset.
:::

:::{grid-item-card} Quick Start on Open Voccabulary Detection
  :link: quick_start/quick_start_ovd.html

  How to use a foundation model in object detection to detect novel classes defined by an unbounded (open) vocabulary.
:::

::::

## Data Preparation

::::{grid} 2
  :gutter: 3

:::{grid-item-card} Prepare COCO2017 Dataset
  :link: data_preparation/prepare_coco17.html

  How to prepare COCO2017 dataset for object detection.
:::

:::{grid-item-card} Prepare Pascal VOC Dataset
  :link: data_preparation/prepare_voc.html

  How to prepare Pascal VOC dataset for object detection.
:::

:::{grid-item-card} Prepare Watercolor Dataset
  :link: data_preparation/prepare_watercolor.html

  How to prepare Watercolor dataset for object detection.
:::

:::{grid-item-card} Convert VOC Format Dataset to COCO Format
  :link: data_preparation/voc_to_coco.html

  How to convert a dataset from VOC format to COCO format for object detection.
:::

:::{grid-item-card} Object Detection with DataFrame
  :link: data_preparation/object_detection_with_dataframe.html

  How to use pd.DataFrame format for object detection
:::
::::

## Finetuning

::::{grid} 2
  :gutter: 3

:::{grid-item-card} Fast Finetune on COCO Format Dataset
  :link: finetune/detection_fast_finetune_coco.html

  How to fast finetune a pretrained model on a dataset in COCO format.
:::

:::{grid-item-card} High Performance Finetune on COCO Format Dataset
  :link: finetune/detection_high_performance_finetune_coco.html

  How to finetune a pretrained model on a dataset in COCO format with high performance.
:::
::::

## Evaulation

::::{grid} 2
  :gutter: 3

:::{grid-item-card} Evaluate Pretrained YOLOv3 on COCO Format Dataset
  :link: evaluation/detection_eval_yolov3_coco.html

  How to evaluate the very fast pretrained YOLOv3 model on dataset in COCO format.
:::

:::{grid-item-card} Evaluate Pretrained Faster R-CNN on COCO Format Dataset
  :link: evaluation/detection_eval_fasterrcnn_coco.html

  How to evaluate the pretrained Faster R-CNN model with high performance on dataset in COCO format.
:::

:::{grid-item-card} Evaluate Pretrained Deformable DETR on COCO Format Dataset
  :link: evaluation/detection_eval_ddetr_coco.html

  How to evaluate the pretrained Deformable DETR model with higher performance on dataset in COCO format
:::

:::{grid-item-card} Evaluate Pretrained Faster R-CNN on VOC Format Dataset
  :link: evaluation/detection_eval_fasterrcnn_voc.html

  How to evaluate the pretrained Faster R-CNN model on dataset in VOC format
:::
::::

```{toctree}
---
maxdepth: 1
hidden: true
---

quick_start/index
evaluation/index
data_preparation/index
finetune/index
```
