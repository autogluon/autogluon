# AutoGluon Multimodal FAQ

## There is no internet access in my deployment environment. What should I do? 

When you have trained the predictor, try to save it with

```python
predictor.save(SAVE_PATH, standalone=True)
```

Afterwards, the following `.load()` call can happen without internet access:

```python
from autogluon.multimodal import MultiModalPredictor

predictor = MultiModalPredictor.load(SAVE_PATH)
```

## What machine is best for running AutoGluon Multimodal?

AutoGluon Multimodal is primarily used for fine-tuning pretrained deep learning models. 
For efficient training, it is highly recommended to utilize GPU machines. 
By default, AutoGluon Multimodal leverages all GPUs available on a single machine. 
However, if training with a single GPU is too slow due to large models or data, it is advisable to switch to a machine with multiple GPUs. 
When using AWS instances, starting with G4 or G5 instances is recommended. 
In cases where GPU memory is insufficient, even with per_gpu_batch_size=1, transitioning to P3dn or P4 instances can be considered.
P4 and P5 instances generally offer superior speed and memory capabilities, but they also come at a higher cost. 
Achieving a balance between performance and cost is essential to find an optimal solution.


## Multi-GPU training encounters RuntimeError: An attempt has been made to start a new process before the current process has finished its bootstrapping phase. How to fix it?

A straightforward solution is to enclose your code within the condition if __name__ == '__main__'. 
By default, we use ddp_spawn as the strategy, which does not fork to initiate the child processes. 
Hence, it is necessary to employ the proper idiom if __name__ == '__main__' in the main module.


## Do I need to preprocess my text or image data before using AutoGluon Multimodal?

Usually you do not need to preprocess the text / image data. AutoGluon Multimodal has built-in 
support of text / image preprocessing. However, this won't block you from appending custom preprocessing logic before 
feeding in the dataframe to AutoGluon Multimodal.


## Is it possible to customize AutoGluon Multimodal?

Yes, check out the [Multimodal Customization](advanced_topics/customization.ipynb) tutorial.

## Can I use AutoGluon Multimodal in Kaggle Competitions?

Yes, we provided a script for building a standalone runnable package of AutoGluon: [AutoGluon Multimodal Kaggle Standalone Package](https://www.kaggle.com/code/linuxdex/get-autogluon-standalone). 
We used this script in our examples about [Petfinder Pawpularity](https://github.com/autogluon/autogluon/tree/master/examples/automm/kaggle_pawpularity) 
and [Feedback Prize - Predicting Effective Arguments](https://github.com/autogluon/autogluon/tree/master/examples/automm/kaggle_feedback_prize). 
You may refer to these examples for more details.

