# Multimodal Prediction

For problems on multimodal data tables that contain image, text, and tabular data, AutoGluon provides `MultiModalPredictor` (abbreviated as `AutoMM`)
that automatically selects, fuses, and tunes foundation models from popular packages like [timm](https://github.com/rwightman/pytorch-image-models),
[huggingface/transformers](https://github.com/huggingface/transformers),
[CLIP](https://github.com/openai/CLIP), [MMDetection](https://github.com/open-mmlab/mmdetection) etc.

You can not only use `AutoMM` to solve standard NLP/Vision tasks
such as sentiment classification, intent detection, paraphrase detection, image classification, but also use it for multimodal problems that involve image,
text, tabular features, object bounding boxes, named entities, etc. Moreover, `AutoMM` can be used as a basic model in the multi-layer
stack-ensemble of [AutoGluon Tabular](https://auto.gluon.ai/stable/tutorials/tabular/index.html), and is powering up the FT-Transformer in `TabularPredictor`.

Here are some example use-cases of AutoMM:

- Multilingual text classification: [Tutorial](text_prediction/multilingual_text)
- Predicting pets' popularity based on their description, photo, and other metadata: [Tutorial](multimodal_prediction/beginner_multimodal), [Example](https://github.com/autogluon/autogluon/tree/master/examples/automm/kaggle_pawpularity)
- Predicting the price of book: [Tutorial](multimodal_prediction/multimodal_text_tabular)
- Scoring student's essays: [Example](https://github.com/autogluon/autogluon/tree/master/examples/automm/kaggle_feedback_prize)
- Image classification: [Tutorial](image_prediction/beginner_image_cls)
- Object detection: [Tutorial](object_detection/quick_start/quick_start_coco), [Example](https://github.com/autogluon/autogluon/tree/master/examples/automm/object_detection)
- Extracting named entities: [Tutorial](text_prediction/ner)
- Search for relevant text / image via text queries: [Tutorial](semantic_matching/index)
- Document Classification (Experimental): [Tutorial](document_prediction/index)


In the following, we decomposed the functionalities of AutoMM and prepared step-by-step guide for each functionality.


## Text Data -- Classification / Regression / NER

::::{grid} 2
  :gutter: 3

:::{grid-item-card} AutoMM for Text Prediction - Quick Start
  :link: text_prediction/beginner_text.html

  How to train high-quality text prediction models with MultiModalPredictor.
:::

:::{grid-item-card} AutoMM for Text Prediction - Multilingual Problems
  :link: text_prediction/multilingual_text.html

  How to use MultiModalPredictor to build models on datasets with languages other than English.
:::

:::{grid-item-card} AutoMM for Named Entity Recognition - Quick Start
  :link: text_prediction/ner.html

  How to use MultiModalPredictor for entity extraction.
:::
::::


## Image Data -- Classification / Regression

::::{grid} 2
  :gutter: 3

:::{grid-item-card} AutoMM for Image Classification - Quick Start
  :link: image_prediction/beginner_image_cls.html

  How to train image classification models with MultiModalPredictor.
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

  How to train high quality object detection model with MultiModalPredictor in under 5 minutes on COCO format dataset.
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
  :link: beginner_semantic_seg.html

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


## Semantic Matching

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

## Multimodal Data -- Classification / Regression

::::{grid} 2
  :gutter: 3

:::{grid-item-card} AutoMM for Text + Tabular - Quick Start
  :link: multimodal_prediction/multimodal_text_tabular.html

  How MultiModalPredictor can be applied to multimodal data tables with a mix of text, numerical, and
  categorical columns.
:::

:::{grid-item-card} AutoMM for Image + Text + Tabular - Quick Start
  :link: multimodal_prediction/beginner_multimodal.html

  How to use MultiModalPredictor to train a model on image, text, numerical, and categorical data.
:::

:::{grid-item-card} AutoMM for Entity Extraction with Text and Image - Quick Start
  :link: multimodal_prediction/multimodal_ner.html

  How to use MultiModalPredictor to train a model for multimodal named entity recognition.
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

:::{grid-item-card} How to use FocalLoss
  :link: advanced_topics/focal_loss.html

  How to use focal loss in AutoMM.
:::

:::{grid-item-card} Faster Prediction with TensorRT
  :link: advanced_topics/tensorrt.html

  How to use TensorRT in accelerating AutoMM model inference.
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
