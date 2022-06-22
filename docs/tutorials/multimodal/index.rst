Multimodal Problems
===================

For problems on multimodal data tables that contain image, text, and tabular data, AutoGluon provides `AutoMMPredictor` that automatically
selects and fuses deep learning backbones from popular packages like [timm](https://github.com/rwightman/pytorch-image-models),
[huggingface/transformers](https://github.com/huggingface/transformers),
[CLIP](https://github.com/openai/CLIP), etc. You can not only use `AutoMMPredictor` to solve classical text and image prediction
problems such as sentiment classification, intent detection, paraphrase detection, image classification,
but also use it on multimodal problems that involve image, text, and tabular features, e.g., predicting the product price
based on the items' description, photo, and other metadata, or matching images with text descriptions. Moreover, `AutoMMPredictor` can
be used as a basic model in the multi-layer stack-ensemble of `TabularPredictor` (Refer to the tutorial ":ref:`sec_tabularprediction_text_multimodal`"  for more details.)

.. container:: cards

   .. card::
      :title: Use AutoGluon Multimodal for Text Prediction: Quick Start
      :link: beginner_text.html

      How to train high-quality text prediction models with AutoMMPredictor in under 5 minutes.

   .. card::
      :title: Solving Multilingual Problems
      :link: multilingual_text.html

      How to use AutoMMPredictor to build models on datasets with languages other than English.

   .. card::
      :title: Multimodal Data Tables with Text
      :link: multimodal_text_tabular.html

      How AutoMMPredictor can be applied to multimodal data tables with a mix of text, numerical, and categorical columns.

   .. card::
      :title: Customize AutoMM
      :link: customization.html

      How to customize AutoMM configurations.


.. toctree::
   :maxdepth: 1
   :hidden:

   beginner_text
   multilingual_text
   multimodal_text_tabular
   customization
