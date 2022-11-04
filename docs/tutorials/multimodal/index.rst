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

.. container:: cards

   .. card::
      :title: AutoMM for Text - Quick Start
      :link: beginner_text.html

      How to train high-quality text prediction models with MultiModalPredictor in under 5 minutes.
    
   .. card::
      :title: AutoMM for Image Classification - Quick Start
      :link: beginner_image_cls.html

      How to train image classification models with MultiModalPredictor.

   .. card::
      :title: AutoMM for Text - Multilingual Problems
      :link: multilingual_text.html

      How to use MultiModalPredictor to build models on datasets with languages other than English.

   .. card::
      :title: AutoMM for Text + Tabular - Quick Start
      :link: multimodal_text_tabular.html

      How MultiModalPredictor can be applied to multimodal data tables with a mix of text, numerical, and categorical columns. Here, we train a model to predict the price of books.

   .. card::
      :title: AutoMM for Multimodal - Quick Start
      :link: beginner_multimodal.html

      How to use MultiModalPredictor to train a model that predicts the adoption speed of pets.

   .. card::
      :title: CLIP in AutoMM - Zero-Shot Image Classification 
      :link: clip_zeroshot.html

      How to use CLIP for zero-shot image classification. 

   .. card::
      :title: CLIP in AutoMM - Extract Embeddings
      :link: clip_embedding.html

      How to use CLIP to extract embeddings for retrieval problem.

   .. card::
      :title: Semantic Search with AutoMM - Extract Embeddings
      :link: semantic_search.html

      How to use semantic embeddings to improve search ranking performance.

   .. card::
      :title: Named Entity Recognition with AutoMM - Quick Start
      :link: ner.html

      How to use AutoMM for entity extraction.

   .. card::
      :title: Single GPU 1B-Scale Model Training with AutoMM via Parameter-Efficient Finetuning
      :link: efficient_finetuning_basic.html

      How to take advantage of larger foundation models with the help of parameter-efficient finetuning.
      In the tutorial, we will use combine IA^3, BitFit, and gradient checkpointing to finetune FLAN-T5-XL.

   .. card::
      :title: Customize AutoMM
      :link: customization.html

      How to customize AutoMM configurations.


.. toctree::
   :maxdepth: 1
   :hidden:

   beginner_text
   beginner_image_cls
   multilingual_text
   multimodal_text_tabular
   beginner_multimodal
   efficient_finetuning_basic
   clip_zeroshot
   clip_embedding
   semantic_search
   ner
   customization
