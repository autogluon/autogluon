# AutoGluon Multimodal (AutoMM): Supercharging Multimodal AutoML with Foundation Models

Foundation models have transformed landscapes across fields like computer vision and natural language processing. 
These models are pre-trained on extensive common-domain data, serving as powerful tools for a wide range of applications. 
However, seamlessly integrating foundation models into real-world application scenarios has posed challenges. 
The diversity of data modalities, the multitude of available foundation models, 
and the considerable model sizes make this integration a nontrivial task.

AutoMM is dedicated to breaking these barriers 
by substantially reducing the engineering effort and manual intervention required in data preprocessing, model selection, and fine-tuning. 
With AutoMM, users can effortlessly adapt foundation models (from popular model zoos like
[HuggingFace](https://github.com/huggingface/transformers), [TIMM](https://github.com/rwightman/pytorch-image-models),
 [MMDetection](https://github.com/open-mmlab/mmdetection)) to their domain-specific data using just three lines of code. 
Our toolkit accommodates various data types, including image, text, tabular, and document data, either individually or in combination. 
It offers support for an array of tasks, encompassing classification, regression, object detection, named entity recognition, semantic matching, and image segmentation.
AutoMM represents a state-of-the-art and user-friendly solution, empowering multimodal AutoML with foundation models. For more details, refer to the paper below:


Zhiqiang, Tang, Haoyang Fang, Su Zhou, Taojiannan Yang, Zihan Zhong, Tony Hu, Katrin Kirchhoff, George Karypis
. ["AutoGluon-Multimodal (AutoMM): Supercharging Multimodal AutoML with Foundation Models"](https://arxiv.org/pdf/2404.16233), The International Conference on Automated Machine Learning (AutoML), 2024.


![AutoMM Introduction](https://automl-mm-bench.s3.amazonaws.com/figures/automm-intro.png)

In the following, we decompose the functionalities of AutoMM and prepare step-by-step guide for each functionality.


## Text Data -- Classification / Regression / NER

::::{grid} 2
  :gutter: 3

:::{grid-item-card} AutoMM for Text Prediction - Quick Start
  :link: text_prediction/beginner_text.html

  How to train high-quality text prediction models with AutoMM.
:::

:::{grid-item-card} AutoMM for Text Prediction - Multilingual Problems
  :link: text_prediction/multilingual_text.html

  How to use AutoMM to build models on datasets with languages other than English.
:::

:::{grid-item-card} AutoMM for Named Entity Recognition - Quick Start
  :link: text_prediction/ner.html

  How to use AutoMM for entity extraction.
:::
::::


## Image Data -- Classification / Regression

::::{grid} 2
  :gutter: 3

:::{grid-item-card} AutoMM for Image Classification - Quick Start
  :link: image_prediction/beginner_image_cls.html

  How to train image classification models with AutoMM.
:::

:::{grid-item-card} Zero-Shot Image Classification with CLIP
  :link: image_prediction/clip_zeroshot.html

  How to enable zero-shot image classification in AutoMM via pretrained CLIP model.
:::
::::


## Image Data -- Object Detection

::::{grid} 2
  :gutter: 3

:::{grid-item-card} Quick Start on a Tiny COCO Format Dataset
  :link: object_detection/quick_start/quick_start_coco.html

  How to train high quality object detection model with AutoMM in under 5 minutes on COCO format dataset.
:::

:::{grid-item-card} Prepare COCO2017 Dataset
  :link: object_detection/data_preparation/prepare_coco17.html

  How to prepare COCO2017 dataset for object detection.
:::

:::{grid-item-card} Prepare Pascal VOC Dataset
  :link: object_detection/data_preparation/prepare_voc.html

  How to prepare Pascal VOC dataset for object detection.
:::

:::{grid-item-card} Prepare Watercolor Dataset
  :link: object_detection/data_preparation/prepare_watercolor.html

  How to prepare Watercolor dataset for object detection.
:::

:::{grid-item-card} Convert VOC Format Dataset to COCO Format
  :link: object_detection/data_preparation/voc_to_coco.html

  How to convert a dataset from VOC format to COCO format for object detection.
:::

:::{grid-item-card} Object Detection with DataFrame
  :link: object_detection/data_preparation/object_detection_with_dataframe.html

  How to use pd.DataFrame format for object detection
:::
::::


## Image Data -- Segmentation

::::{grid} 2
  :gutter: 3

:::{grid-item-card} AutoMM for Semantic Segmentation - Quick Start
  :link: image_segmentation/beginner_semantic_seg.html

  How to train semantic segmentation models with AutoMM.
:::
::::


## Document Data -- Classification / Regression

::::{grid} 2
  :gutter: 3

:::{grid-item-card} AutoMM for Scanned Document Classification
  :link: document_prediction/document_classification.html

  How to use AutoMM to build a scanned document classifier.
:::

:::{grid-item-card} Classifying PDF Documents with AutoMM
  :link: document_prediction/pdf_classification.html

  How to use AutoMM to build a PDF document classifier.
:::
::::


## Image / Text Data -- Semantic Matching

::::{grid} 2
  :gutter: 3

:::{grid-item-card} Text-to-text Semantic Matching with AutoMM - Quick Start
  :link: semantic_matching/text2text_matching.html

  How to use AutoMM for text-to-text semantic matching.
:::

:::{grid-item-card} Image-to-Image Semantic Matching with AutoMM - Quick Start
  :link: semantic_matching/image2image_matching.html

  How to use AutoMM for image-to-image semantic matching.
:::

:::{grid-item-card} Image-Text Semantic Matching with AutoMM - Quick Start
  :link: semantic_matching/image_text_matching.html

  How to use AutoMM for image-text semantic matching.
:::

:::{grid-item-card} Zero Shot Image-Text Semantic Matching with AutoMM
  :link: semantic_matching/zero_shot_img_txt_matching.html

  How to use AutoMM for zero shot image-text semantic matching.
:::

:::{grid-item-card} Text Semantic Search with AutoMM
  :link: semantic_matching/text_semantic_search.html

  How to use semantic embeddings to improve search ranking performance.
:::

::::

## Multimodal Data -- Classification / Regression / NER

::::{grid} 2
  :gutter: 3

:::{grid-item-card} AutoMM for Text + Tabular - Quick Start
  :link: multimodal_prediction/multimodal_text_tabular.html

  How AutoMM can be applied to multimodal data tables with a mix of text, numerical, and
  categorical columns.
:::

:::{grid-item-card} AutoMM for Image + Text + Tabular - Quick Start
  :link: multimodal_prediction/beginner_multimodal.html

  How to use AutoMM to train a model on image, text, numerical, and categorical data.
:::

:::{grid-item-card} AutoMM for Entity Extraction with Text and Image - Quick Start
  :link: multimodal_prediction/multimodal_ner.html

  How to use AutoMM to train a model for multimodal named entity recognition.
:::
::::


## Advanced Topics

::::{grid} 2
  :gutter: 3

:::{grid-item-card} Single GPU Billion-scale Model Training via Parameter-Efficient Finetuning
  :link: advanced_topics/efficient_finetuning_basic.html

  How to take advantage of larger foundation models with the help of parameter-efficient finetuning.
  In the tutorial, we will use combine IA^3, BitFit, and gradient checkpointing to finetune FLAN-T5-XL.
:::

:::{grid-item-card} Hyperparameter Optimization in AutoMM
  :link: advanced_topics/hyperparameter_optimization.html

  How to do hyperparameter optimization in AutoMM.
:::

:::{grid-item-card} Knowledge Distillation in AutoMM
  :link: advanced_topics/model_distillation.html

  How to do knowledge distillation in AutoMM.
:::

:::{grid-item-card} Continuous Training with AutoMM
  :link: advanced_topics/continuous_training.html

  How to continue training in AutoMM.
:::

:::{grid-item-card} Customize AutoMM
  :link: advanced_topics/customization.html

  How to customize AutoMM configurations.
:::

:::{grid-item-card} AutoMM Presets
  :link: advanced_topics/presets.html

  How to use AutoMM presets.
:::

:::{grid-item-card} Few Shot Learning with AutoMM
  :link: advanced_topics/few_shot_learning.html

  How to use foundation models + SVM for few shot learning.
:::

:::{grid-item-card} Handling Class Imbalance with AutoMM - Focal Loss
  :link: advanced_topics/focal_loss.html

  How to use AutoMM to handle class imbalance.
:::

:::{grid-item-card} Faster Prediction with TensorRT
  :link: advanced_topics/tensorrt.html

  How to use TensorRT in accelerating AutoMM model inference.
:::

:::{grid-item-card} AutoMM Problem Types and Evaluation Metrics.
  :link: advanced_topics/problem_types_and_metrics.html

  A comprehensive guide to AutoGluon's supported problem types and their evaluation metrics.
:::

:::{grid-item-card} Multiple Label Columns
  :link: advanced_topics/multiple_label_columns.html

  How to handle multiple label columns with AutoGluon MultiModal.
:::

::::

```{toctree}
---
caption: Multimodal Prediction
maxdepth: 1
hidden: true
---

multimodal_prediction/index
```

```{toctree}
---
caption: Object Detection
maxdepth: 2
hidden: true
---

object_detection/index
```

```{toctree}
---
caption: Image Prediction
maxdepth: 1
hidden: true
---

image_prediction/index
```

```{toctree}
---
caption: Image Segmentation
maxdepth: 1
hidden: true
---

image_segmentation/index
```

```{toctree}
---
caption: Text Prediction
maxdepth: 1
hidden: true
---

text_prediction/index
```

```{toctree}
---
caption: Document Prediction
maxdepth: 1
hidden: true
---

document_prediction/index
```

```{toctree}
---
caption: Semantic Matching
maxdepth: 1
hidden: true
---

semantic_matching/index
```

```{toctree}
---
caption: Advanced Topics
maxdepth: 1
hidden: true
---

advanced_topics/index
```
