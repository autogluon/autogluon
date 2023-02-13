Multimodal Prediction
=====================

For problems on multimodal data tables that contain image, text, and tabular data, AutoGluon provides `MultiModalPredictor` (abbreviated as `AutoMM`)
that automatically selects, fuses, and tunes foundation models from popular packages like `timm <https://github.com/rwightman/pytorch-image-models>`_,
`huggingface/transformers <https://github.com/huggingface/transformers>`_,
`CLIP <https://github.com/openai/CLIP>`_, `MMDetection <https://github.com/open-mmlab/mmdetection>`_ etc.

You can not only use `AutoMM` to solve standard NLP/Vision tasks
such as sentiment classification, intent detection, paraphrase detection, image classification, but also use it for multimodal problems that involve image,
text, tabular features, object bounding boxes, named entities, etc. Moreover, `AutoMM` can be used as a basic model in the multi-layer
stack-ensemble of `AutoGluon Tabular <https://auto.gluon.ai/stable/tutorials/tabular_prediction/index.html>`_, and is powering up the FT-Transformer in `TabularPredictor`.

Here are some example use-cases of AutoMM:

- Multilingual text classification :doc:`[Tutorial] <text_prediction/multilingual_text>`
- Predicting pets' popularity based on their description, photo, and other metadata. :doc:`[Tutorial] <multimodal_prediction/beginner_multimodal>` `[Example] <https://github.com/autogluon/autogluon/tree/master/examples/automm/kaggle_pawpularity>`_.
- Predicting the price of book. :doc:`[Tutorial] <multimodal_prediction/multimodal_text_tabular>`.
- Scoring student's essays. `[Example] <https://github.com/autogluon/autogluon/tree/master/examples/automm/kaggle_feedback_prize>`_.
- Image classification. :doc:`[Tutorial] <image_prediction/beginner_image_cls>`.
- Object detection. :doc:`[Tutorial] <object_detection/quick_start/quick_start_coco>` `[Example] <https://github.com/autogluon/autogluon/tree/master/examples/automm/object_detection>`_.
- Extracting named entities. :doc:`[Tutorial] <text_prediction/ner>`.
- Search for relevant text / image via text queries. :doc:`[Tutorial] <matching/index>`.


In the following, we decomposed the functionalities of AutoMM and prepared step-by-step guide for each functionality.


Text Data
---------
.. container:: cards

   .. card::
      :title: AutoMM for Text Prediction - Quick Start
      :link: text_prediction/beginner_text.html

      How to train high-quality text prediction models with MultiModalPredictor.

   .. card::
      :title: AutoMM for Text Prediction - Multilingual Problems
      :link: text_prediction/multilingual_text.html

      How to use MultiModalPredictor to build models on datasets with languages other than English.

   .. card::
      :title: Named Entity Recognition with AutoMM - Quick Start
      :link: text_prediction/ner.html

      How to use MultiModalPredictor for entity extraction.


Image Data -- Classification / Regression
-----------------------------------------
.. container:: cards

   .. card::
      :title: AutoMM for Image Classification - Quick Start
      :link: image_prediction/beginner_image_cls.html

      How to train image classification models with MultiModalPredictor.

   .. card::
      :title: Zero-Shot Image Classification with CLIP
      :link: image_prediction/clip_zeroshot.html

      How to enable zero-shot image classification in AutoMM via pretrained CLIP model.


Image Data -- Object Detection
--------------------------
.. container:: cards

   .. card::
      :title: Quick Start on a Tiny COCO Format Dataset
      :link: object_detection/quick_start/quick_start_coco.html

      How to train high quality object detection model with MultiModalPredictor in under 5 minutes on COCO format dataset.

   .. card::
      :title: Prepare COCO2017 Dataset
      :link: object_detection/data_preparation/prepare_coco17.html

      How to prepare COCO2017 dataset for object detection.

   .. card::
      :title: Prepare Pascal VOC Dataset
      :link: object_detection/data_preparation/prepare_voc.html

      How to prepare Pascal VOC dataset for object detection.

   .. card::
      :title: Prepare Watercolor Dataset
      :link: object_detection/data_preparation/prepare_watercolor.html

      How to prepare Watercolor dataset for object detection.

   .. card::
      :title: Convert VOC Format Dataset to COCO Format
      :link: object_detection/data_preparation/voc_to_coco.html

      How to convert a dataset from VOC format to COCO format for object detection.

   .. card::
      :title: Fast Finetune on COCO Format Dataset
      :link: object_detection/finetune/detection_fast_finetune_coco.html

      How to fast finetune a pretrained model on a dataset in COCO format.

   .. card::
      :title: High Performance Finetune on COCO Format Dataset
      :link: object_detection/finetune/detection_high_performance_finetune_coco.html

      How to finetune a pretrained model on a dataset in COCO format with high performance.

   .. card::
      :title: Evaluate Pretrained YOLOv3 on COCO Format Dataset
      :link: object_detection/evaluation/detection_eval_yolov3_coco.html

      How to evaluate the very fast pretrained YOLOv3 model on dataset in COCO format.

   .. card::
      :title: Evaluate Pretrained Faster R-CNN on COCO Format Dataset
      :link: object_detection/evaluation/detection_eval_fasterrcnn_coco.html

      How to evaluate the pretrained Faster R-CNN model with high performance on dataset in COCO format.

   .. card::
      :title: Evaluate Pretrained Deformable DETR on COCO Format Dataset
      :link: object_detection/evaluation/detection_eval_ddetr_coco.html

      How to evaluate the pretrained Deformable DETR model with higher performance on dataset in COCO format

   .. card::
      :title: Evaluate Pretrained Faster R-CNN on VOC Format Dataset
      :link: object_detection/evaluation/detection_eval_fasterrcnn_voc.html

      How to evaluate the pretrained Faster R-CNN model on dataset in VOC format


Document Data
-------------
.. container:: cards

   .. card::
      :title: AutoMM for Scanned Document Classification
      :link: document/document_classification.html

      How to use MultiModalPredictor to build a scanned document classifier.


Matching
--------
.. container:: cards

   .. card::
      :title: Text-to-text Matching with AutoMM - Quick Start
      :link: matching/text2text_matching.html

      How to use AutoMM for text to text matching.

   .. card::
      :title: Semantic Textual Search with AutoGluon Multimodal Matching
      :link: matching/semantic_search.html

      How to use semantic embeddings to improve search ranking performance.

   .. card::
      :title: Extract Image/Text Embeddings in AutoMM for Matching Problems
      :link: matching/clip_embedding.html

      How to use CLIP to extract embeddings for retrieval problem.


Multimodal Data
---------------
.. container:: cards

   .. card::
      :title: AutoMM for Text + Tabular - Quick Start
      :link: multimodal_prediction/multimodal_text_tabular.html

      How MultiModalPredictor can be applied to multimodal data tables with a mix of text, numerical, and
      categorical columns. Here, we train a model to predict the price of books.

   .. card::
      :title: AutoMM for Image + Text + Tabular - Quick Start
      :link: multimodal_prediction/beginner_multimodal.html

      How to use MultiModalPredictor to train a model that predicts the adoption speed of pets.

   .. card::
      :title: AutoMM for Entity Extraction with Text and Image - Quick Start
      :link: multimodal_prediction/multimodal_ner.html

      How to use MultiModalPredictor to train a model for multimodal named entity recognition.


Advanced Topics
---------------
.. container:: cards

   .. card::
      :title: Single GPU Billion-scale Model Training via Parameter-Efficient Finetuning
      :link: advanced_topics/efficient_finetuning_basic.html

      How to take advantage of larger foundation models with the help of parameter-efficient finetuning.
      In the tutorial, we will use combine IA^3, BitFit, and gradient checkpointing to finetune FLAN-T5-XL.

   .. card::
      :title: HPO in AutoMM
      :link: advanced_topics/hyperparameter_optimization.html

      How to do hyperparameter optimization in AutoMM.

   .. card::
      :title: Knowledge Distillation in AutoMM
      :link: advanced_topics/model_distillation.html

      How to do knowledge distillation in AutoMM.

   .. card::
      :title: Customize AutoMM
      :link: advanced_topics/customization.html

      How to customize AutoMM configurations.

   .. card::
      :title: AutoMM Presets
      :link: presets.html

      How to use AutoMM presets.

FAQ
---
.. container:: cards

   .. card::
      :title: Frequently Asked Questions
      :link: advanced_topics/multimodal_faq.html

      Frequently Asked Questions


.. toctree::
   :maxdepth: 2
   :hidden:

   text_prediction/index
   image_prediction/index
   object_detection/index
   document/index
   matching/index
   multimodal_prediction/index
   advanced_topics/index
   advanced_topics/multimodal_faq.md
