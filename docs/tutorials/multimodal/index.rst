Multimodal Prediction
=====================

For problems on multimodal data tables that contain image, text, and tabular data, AutoGluon provides `AutoMMPredictor` (or simply calling it AutoMM)
that automatically selects and fuses deep learning backbones from popular packages like `timm <https://github.com/rwightman/pytorch-image-models>`_,
`huggingface/transformers <https://github.com/huggingface/transformers>`_,
`CLIP <https://github.com/openai/CLIP>`_, etc. You can not only use `AutoMMPredictor` to solve classical text and image prediction
problems such as sentiment classification, intent detection, paraphrase detection, image classification,
but also use it on multimodal problems that involve image, text, and tabular features, e.g., predicting the product price
based on the items' description, photo, and other metadata, or matching images with text descriptions. Moreover, `AutoMMPredictor` can
be used as a basic model in the multi-layer stack-ensemble of `TabularPredictor`.

.. container:: cards

   .. card::
      :title: AutoMM for Text - Quick Start
      :link: beginner_text.html

      How to train high-quality text prediction models with AutoMMPredictor in under 5 minutes.
    
   .. card::
      :title: AutoMM for Image Classification - Quick Start
      :link: beginner_image_cls.html

      How to train image classfiers with AutoMMPredictor.

   .. card::
      :title: AutoMM for Text - Multilingual Problems
      :link: multilingual_text.html

      How to use AutoMMPredictor to build models on datasets with languages other than English.

   .. card::
      :title: AutoMM for Text + Tabular - Quick Start
      :link: multimodal_text_tabular.html

      How AutoMMPredictor can be applied to multimodal data tables with a mix of text, numerical, and categorical columns.

   .. card::
      :title: AutoMM for Multimodal - Quick Start
      :link: beginner_multimodal.html

      How to train high-quality multimodal models with AutoMMPredictor.

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
   customization
