# FAQ
:label:`sec_multimodal_faq`


### There is no internet access in my deployment environment. What should I do? 

When you have trained the predictor, try to save it with

```python
predictor.save(SAVE_PATH, standalone=True)
```

Afterwards, the following `.load()` call can happen without internet access:

```python
from autogluon.multimodal import MultiModalPredictor

predictor = MultiModalPredictor.load(SAVE_PATH)
```


### Do I need to preprocess my text or image data before using AutoGluon Multimodal?

Usually you do not need to preprocess the text / image data. AutoGluon Multimodal has built-in 
support of text / image preprocessing. However, this won't block you from appending custom preprocessing logic before 
feeding in the dataframe to AutoGluon Multimodal.


### Is it possible to customize AutoGluon Multimodal?

Yes, check our tutorial at :ref:`sec_automm_customization`. 

### Can I use AutoGluon Multimodal in Kaggle Competitions?

Yes, we provided a script for building a standalone runnable package of AutoGluon: [AutoGluon Multimodal Kaggle Standalone Package](https://www.kaggle.com/code/linuxdex/get-autogluon-standalone). 
We used this script in our examples about [Petfinder Pawpularity](https://github.com/autogluon/autogluon/tree/master/examples/automm/kaggle_pawpularity) 
and [Feedback Prize - Predicting Effective Arguments](https://github.com/autogluon/autogluon/tree/master/examples/automm/kaggle_feedback_prize). 
You may refer to these examples for more details.

### 
