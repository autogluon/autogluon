# Use MultiModal Feature Extraction to Create a Few-shot Memory Bank Model

[memory_bank.py](./memory_bank.py): This example provides a simple and clear way to implement a few-shot memory bank model with AutoGluon MultiModal Feature Extraction according to [Tip-Adapter](https://github.com/gaopengcuhk/Tip-Adapter) [1]. 

### 1. Run Example

Before running the example, the image datasets need to be prepared with additional code. We follow the few-shot dataset design in [CoOP](https://github.com/KaiyangZhou/CoOp). Copy folder `configs` and `datasets` from the repository and rename `datasets` to `imagedatasets` to avoid conflict with `datasets` in Hugging Face. You can download the datasets and the splits under the [direction](https://github.com/KaiyangZhou/CoOp/blob/main/DATASETS.md).

For text datasets, datasets in Hugging Face are supported. We recommend datasets under [SetFit](https://huggingface.co/datasets?sort=downloads&search=SetFit) [2] for few-shot learning. Remember to change the column names in args if the data column is not named `text` or have multi-text information.

After preparing the datasets, run the example:

    python memory_bank.py --type clip --dataset food101 --shots 16

- `type` is the type of few-shot learning. You can choose from `clip`, `text`, `image` for different backbone methods.
- `backbone` is the backbone model of MultiModal Predictor.
- `data_path` is the path for image dataset.
- `dataset` is the name of dataset. Support image datasets in COOP and text datasets in Hugging Face.
- `column_names` is the names of the data columns.
- `label_column` is the name of the label column.
- `shots` is the shots for each class in training set.
- `aug_epochs` is the epochs to create the bank.
- `model_head_type` is the model head for few-shot classification. You can choose from 'linear' and 'SVM' for different classification heads.
- `lr` is the learning rate for training the model head.
- `lr_F` is the learning rate for finetuing the memory bank.
- `train_epoch` is the training epochs for training the model head.
- `train_epoch_F` is the training epochs for fine-tuning the memory bank.
- `init_alpha` is initial values of hyper-parameters in memory bank. `alpha` adjusts the weight of probability between the classifier and memory bank.
- `init_beta` is initial values of hyper-parameters in memory bank. `beta` modulates the sharpness when converting the similarities into non-negative values.
- `search_scale` is the search scale of alpha and beta.
- `search_step` is the steps of searching hyper-parameters alpha and beta.

### 2. Method

Memory bank follows the excellent design of [Tip-Adapter](https://arxiv.org/pdf/2207.09519.pdf) which stores the image features of few-shot training set to improve the performance of zero-shot CLIP through feature similarity. The stored features can also serve as the initialization of a trainable classifier. This ProtoNet-like design makes full use of few-shot training information and leads to good performance [3]. We believe that the effectiveness of this design is not limited to CLIP, and can be widely applied to few-shot classification tasks of images and texts. 

Memory bank which is the derivative application of Tip-Adapter obtains diversified multi-modal features through MultiModal Feature Extraction. In this example, we first trained a linear classifier based on multi-modal features to generate baseline accuracy. Then, the similarity result between features and memory bank is introduced to baseline predict probability. Finally, an additional linear adapter which is initialized with memory bank is trained to help few-shot classification.

Hyper-parameters `alpha` and `beta` which adjust the memory bank are modified through grid search on validation set to attain the superior performance.

### 3. Training Results

The training results of memory bank can be seen in the followed table. `Acc w/o memory bank` is the few-shot classification accuracy of method. `ACC w. memory bank` and `ACC w. memory bank (+finetune)` show the classification results after introducing and fine-tuning the memory bank, respectively.

| Datasets | Method | BackBone | Head | Shots | lr | lr_F | Acc w/o memory bank| Acc w. memory bank | Acc w. memory bank (+finetune) | 
|----------|--------|----------|------|-------|----|------|--------------------|--------------------|--------------------------------| 
| Food-101 | CLIP | openai/clip-vit-large-patch14-336 | NaN | 16 | NaN | 1e-3 | 91.90 | 92.42 | 92.80 | 
| Food-101 | CLIP | openai/clip-vit-large-patch14-336 | NaN | 1 | NaN | 1e-3 | 91.90 | 91.99 | 91.97 | 
| Food-101 | CLIP | openai/clip-vit-large-patch14-336 | NaN | 64 | NaN | 1e-3 | 91.90 | 92.43 | 93.10 | 
| Food-101 | CLIP | openai/clip-vit-base-patch32 | NaN | 16 | NaN | 1e-3 | 80.42 | 80.88 | 82.01 | 
| Caltech101 | CLIP | openai/clip-vit-large-patch14-336 | NaN | 16 | NaN | 1e-3 | 94.48 | 97.32 | 98.80 | 
| DTD | CLIP | openai/clip-vit-large-patch14-336 | NaN | 16 | NaN | 1e-3 | 54.2 | 69.86 | 72.10 | 
| EuraSAT | CLIP | openai/clip-vit-large-patch14-336 | NaN | 16 | NaN | 1e-3 | 61.48 | 79.01 | 83.65 | 
| Flower-102 | CLIP | openai/clip-vit-large-patch14-336 | NaN | 16 | NaN | 1e-3 | 79.13 | 96.95 | 96.51 | 
| Oxford Pets | CLIP | openai/clip-vit-large-patch14-336 | NaN | 16 | NaN | 1e-3 | 93.79 | 94.22 | 95.52 | 
| Stanford Cars | CLIP | openai/clip-vit-large-patch14-336 | NaN | 16 | NaN | 1e-3 | 78.20 | 84.09 | 87.95 | 
| Food-101 | Image | swin_base_patch4_window7_224 | Linear | 16 | 1e-2 | 1e-3 | 73.66 | 73.64 | 76.18 | 
| Caltech101 | Image | swin_base_patch4_window7_224 | Linear | 16 | 1e-2 | 1e-3 | 96.75 | 96.75 | 97.16 | 
| DTD | Image | swin_base_patch4_window7_224 | Linear | 16 | 1e-2 | 1e-3 | 67.55 | 68.26 | 70.45 | 
| SetFit/sst5 | Text | sentence-transformers/paraphrase-mpnet-base-v2 | Linear | 16 | 1e-2 | 1e-3 | 38.42 | 38.42 | 39.23 | 
| SetFit/sst5 | Text | sentence-transformers/paraphrase-mpnet-base-v2 | Linear | 1 | 1e-2 | 1e-3 | 33.08 | 33.08 | 33.08 | 
| SetFit/sst5 | Text | sentence-transformers/paraphrase-mpnet-base-v2 | Linear | 64 | 1e-2 | 1e-3 | 45.61 | 46.02 | 48.19 | 
| SetFit/sst5 | Text | sentence-transformers/msmarco-MiniLM-L-12-v3 | Linear | 16 | 1e-2 | 1e-3 | 30.18 | 30.86 | 30.59 | 
| SetFit/Emotion | Text | sentence-transformers/paraphrase-mpnet-base-v2 | Linear | 16 | 1e-2 | 1e-3 | 43.10 | 43.65 | 43.90 | 
| SetFit/subj | Text | sentence-transformers/paraphrase-mpnet-base-v2 | Linear | 16 | 1e-2 | 1e-3 | 90.50 | 90.55 | 90.75 | 
| SetFit/20_newsgroups | Text | sentence-transformers/paraphrase-mpnet-base-v2 | Linear | 16 | 1e-2 | 1e-3 | 54.14 | 57.36 | 58.90 | 
| SetFit/enron_spam | Text | sentence-transformers/paraphrase-mpnet-base-v2 | Linear | 16 | 1e-2 | 1e-3 | 91.35 | 91.70 | 92.85 | 
| SetFit/SentEval-CR | Text | sentence-transformers/paraphrase-mpnet-base-v2 | Linear | 16 | 1e-2 | 1e-3 | 88.31 | 88.58 | 89.24 | 
| Food-101 | Image | swin_base_patch4_window7_224 | SVM | 16 | NaN | 1e-3 | 73.06 | 74.42 | 75.72 | 
| Caltech101 | Image | swin_base_patch4_window7_224 | SVM | 16 | NaN | 1e-3 | 93.10 | 97.16 | 97.44 | 
| DTD | Image | swin_base_patch4_window7_224 | SVM | 16 | NaN | 1e-3 | 69.39 | 70.45 | 70.39 | 
| SetFit/sst5 | Text | sentence-transformers/paraphrase-mpnet-base-v2 | SVM | 16 | NaN | 1e-3 | 30.90 | 39.28 | 39.95 | 
| SetFit/Emotion | Text | sentence-transformers/paraphrase-mpnet-base-v2 | SVM | 16 | NaN | 1e-3 | 26.55 | 43.15 | 44.20 | 
| SetFit/20_newsgroups | Text | sentence-transformers/paraphrase-mpnet-base-v2 | SVM | 16 | NaN | 1e-3 | 48.43 | 57.90 | 58.72 | 


---

### Reference

[1] Tip-Adapter: Training-free CLIP-Adapter for Better Vision-Language Modeling. <https://arxiv.org/pdf/2207.09519.pdf>

[2] Efficient Few-Shot Learning Without Prompts. <https://arxiv.org/pdf/2209.11055.pdf>

[3] Prototypical Networks for Few-shot Learning. <https://papers.nips.cc/paper/2017/file/cb8da6767461f2812ae4290eac7cbc42-Paper.pdf>
