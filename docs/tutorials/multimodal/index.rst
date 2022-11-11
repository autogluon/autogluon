Multimodal Prediction
=====================

For problems on multimodal data tables that contain image, text, and tabular data, AutoGluon provides `MultiModalPredictor` (abbreviated as `AutoMM`)
that automatically selects and fuses deep learning backbones from popular packages like `timm <https://github.com/rwightman/pytorch-image-models>`_,
`huggingface/transformers <https://github.com/huggingface/transformers>`_,
`CLIP <https://github.com/openai/CLIP>`_, etc. You can use it to build models for multimodal problems that involve image, text, and tabular features, e.g., predicting the product price
based on the items' description, photo, and other metadata, or matching images with text descriptions.

In addition, being good at multimodal problems implies that the predictor will be good for **each specific modality**. Thus, you can also use `AutoMM` to solve standard NLP/Vision tasks like sentiment classification,
intent detection, paraphrase detection, image classification. Moreover, `AutoMM` can be used as a basic model in the multi-layer stack-ensemble of `TabularPredictor`.

In the following, we prepared a few tutorials to help you learn how to use `AutoMM` to solve problems that involve image, text, and tabular data.


Text Prediction and Entity Extraction
-------------------------------------
.. container:: cards

   .. card::
      :title: AutoMM for Text Prediction - Quick Start
      :link: text_prediction/beginner_text.html

      How to train high-quality text prediction models with MultiModalPredictor in under 5 minutes.

   .. card::
      :title: AutoMM for Text Prediction - Multilingual Problems
      :link: text_prediction/multilingual_text.html

      How to use MultiModalPredictor to build models on datasets with languages other than English.

   .. card::
      :title: Named Entity Recognition with AutoMM - Quick Start
      :link: text_prediction/ner.html

      How to use MultiModalPredictor for entity extraction.


Image Prediction
----------------
.. container:: cards

   .. card::
      :title: AutoMM for Image Classification - Quick Start
      :link: image_prediction/beginner_image_cls.html

      How to train image classification models with MultiModalPredictor.

   .. card::
      :title: Zero-Shot Image Classification with CLIP
      :link: image_prediction/clip_zeroshot.html

      How to enable zero-shot image classification in AutoMM via pretrained CLIP model.


Object Detection
----------------
.. container:: cards



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


Multimodal Classification / Regression
--------------------------------------
.. container:: cards

   .. card::
      :title: AutoMM for Text + Tabular - Quick Start
      :link: mulitmodal_prediction/multimodal_text_tabular.html

      How MultiModalPredictor can be applied to multimodal data tables with a mix of text, numerical, and
      categorical columns. Here, we train a model to predict the price of books.

   .. card::
      :title: AutoMM for Image + Text + Tabular - Quick Start
      :link: mulitmodal_prediction/beginner_multimodal.html

      How to use MultiModalPredictor to train a model that predicts the adoption speed of pets.


Advanced Topics
---------------
.. container:: cards

   .. card::
      :title: Single GPU Billion-scale Model Training via Parameter-Efficient Finetuning
      :link: advanced_topics/efficient_finetuning_basic.html

      How to take advantage of larger foundation models with the help of parameter-efficient finetuning.
      In the tutorial, we will use combine IA^3, BitFit, and gradient checkpointing to finetune FLAN-T5-XL.
 
   .. card::
      :title: Customize AutoMM
      :link: advanced_topics/customization.html

      How to customize AutoMM configurations.


.. toctree::
   :maxdepth: 2
   :hidden:

   text_prediction/index
   image_prediction/index
   matching/index
   multimodal_prediction/index
   object_detection/index
   advanced_topics/index
