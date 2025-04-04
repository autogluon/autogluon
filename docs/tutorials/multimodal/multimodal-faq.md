# AutoGluon Multimodal FAQ

## How does AutoGluon MultiModal model multimodal data?
We can automatically detect image, text, tabular, document, and any of their combinations from your data.
For each modality, we select a corresponding foundation model. 
If more than one modality are detected, a fusion module will be created on top of the unimodal backbones to fuse their features and make final predictions, hence late fusion.

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
When using AWS instances, starting with [G4](https://aws.amazon.com/ec2/instance-types/g4/) or [G5](https://aws.amazon.com/ec2/instance-types/g5/) instances is recommended. 
In cases where GPU memory is insufficient, even with per_gpu_batch_size=1, transitioning to [p3dn.24xlarge](https://aws.amazon.com/ec2/instance-types/p3/) or [P4](https://aws.amazon.com/ec2/instance-types/p4/) instances can be considered.
[P4](https://aws.amazon.com/ec2/instance-types/p4/) and [P5](https://press.aboutamazon.com/2023/3/aws-and-nvidia-collaborate-on-next-generation-infrastructure-for-training-large-machine-learning-models-and-building-generative-ai-applications) instances generally offer superior speed and memory capabilities, but they also come at a higher cost. 
Achieving a balance between performance and cost is essential to find an optimal solution.


## Multi-GPU training encounters RuntimeError: An attempt has been made to start a new process before the current process has finished its bootstrapping phase. How to fix it?

A straightforward solution is to enclose your code within the condition `if __name__ == '__main__'`. 
By default, we use ddp_spawn as the strategy, which does not fork to initiate the child processes. 
Hence, it is necessary to employ the proper idiom `if __name__ == '__main__'` in the main module.


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

## How does AutoGluon MultiModal handle multiple images per sample?

We provide two options:
1. Utilizing multiple image columns in a dataframe.
2. Concatenating multiple image paths with semicolons and putting them into one dataframe column.

These options can be used individually or in combination. All the image columns are automatically detected.
During processing, we pass all images through a single image backbone and average their features and logits to obtain the final representation or prediction. 
Note that for option 2, the maximum number of images allowed for one dataframe column can be controlled via hyperparameter `timm_image.max_img_num_per_col`, which has a default value of 2.

## How does AutoGluon MultiModal handle multiple text columns in a dataframe?

The detection of all text columns is automated.
We tokenize each text field individually and then concatenate them into a single token sequence before feeding it into the model. 
The order of the text columns in the concatenated sequence follows the order of `sorted(X.columns)`, where `X` represents the dataframe.
The maximum length of the token sequence is determined by the hyperparameter `hf_text.max_text_len`, with a default value of 512. 
If the sequence length exceeds this limit, we perform iterative truncation by removing one token from the tail of the longest text field in each iteration. 
This approach ensures that shorter text fields are less likely to be affected during the sequence truncation process.
Furthermore, you can control the inclusion of separation tokens among different text fields by adjusting the hyperparameter `hf_text.insert_sep`, which is set to True by default.
