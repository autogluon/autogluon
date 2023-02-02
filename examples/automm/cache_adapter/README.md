# Use MultiModal Feature Extraction to Create a Few-shot Cache Adapter Model

[cache_adapter.py](./cache_adapter.py): This example provides a simple and clear way to implement a few-shot cache adapter model with AutoGluon MultiModal Feature Extraction according to [Tip-adapter](https://github.com/gaopengcuhk/Tip-Adapter) [1]. 

### 1. Run Example

Before running the example, the image datasets need to be prepared with additional code. We follow the few-shot dataset design in [CoOP](https://github.com/KaiyangZhou/CoOp). Copy folder `configs` and `datasets` from the repository and rename `datasets` to `imagedatasets` to avoid conflict with `datasets` in Hugging Face. You can download the datasets and the splits under the [direction]().

For text datasets, datasets in Hugging Face are supported. We recommend datasets under SetFit for few-shot learning. Remember to change the column names in args if the data column is not named `text` or have multi-text information.

After preparing the datasets, run the example:

    python cache_adapter.py --type clip --dataset food101 --shots 16

- `type` is the type of few-shot learning. You choose from `clip`, `text`, `image` for different backbone methods.
- `backbone` is the backbone model of MultiModal Predictor.
- `data_path` is the path for image dataset.
- `dataset` is the name of dataset. Support image datasets in COOP and text datasets in Hugging Face.
- `column_names` is the names of the data columns.
- `label_column` is the name of the label column.
- `shots` is the shots for each class in training set.
- `aug_epochs` is the epochs to create the cache.
- `lr` is the learning rate for training the model head.
- `lr_F` is the learning rate for finetuing the cache adapter.
- `train_epoch` is the training epochs for training the model head.
- `train_epoch_F` is the training epochs for fine-tuning the cache adapter.
- `init_alpha` and `init_beta` are initial values of hyper-parameters in cache adapter.
- `search_scale` is the search scale of alpha and beta.
- `search_step` is the steps of searching hyper-parameters alpha and beta.

### 2. Training Results

The training results of cache adapter can be seen in the followed table. `Acc` is the few-shot classification accuracy of method. `Cache Adapter ACC` and `Finetuned Cache Adapter ACC` show the classification results after introducing and fine-tuning the cache adapter, respectively.

| Datasets | Method | BackBone | Shots | lr | lr_F | Acc | Cache Adapter Acc | Fine-tuned Cache Adapter Acc| 
|----------|--------|----------|-------|----|------|-----|-------------------|-----------------------------| 
| Food-101 | CLIP | openai/clip-vit-large-patch14-336 | 16 | NaN | 1e-3 | 91.90 | 92.42 | 92.80 | 
| Food-101 | CLIP | openai/clip-vit-large-patch14-336 | 1 | NaN | 1e-3 | 91.90 | 91.99 | 91.97 | 
| Food-101 | CLIP | openai/clip-vit-large-patch14-336 | 64 | NaN | 1e-3 | 91.90 | 92.43 | 93.10 | 
| Food-101 | CLIP | openai/clip-vit-base-patch32 | 16 | NaN | 1e-3 | 80.42 | 80.88 | 82.01 | 
| Caltech101 | CLIP | openai/clip-vit-large-patch14-336 | 16 | NaN | 1e-3 | 94.48 | 97.32 | 98.80 | 
| DTD | CLIP | openai/clip-vit-large-patch14-336 | 16 | NaN | 1e-3 | 54.2 | 69.86 | 72.10 | 
| EuraSAT | CLIP | openai/clip-vit-large-patch14-336 | 16 | NaN | 1e-3 | 61.48 | 79.01 | 83.65 | 
| Flower-102 | CLIP | openai/clip-vit-large-patch14-336 | 16 | NaN | 1e-3 | 79.13 | 96.95 | 96.51 | 
| Oxford Pets | CLIP | openai/clip-vit-large-patch14-336 | 16 | NaN | 1e-3 | 93.79 | 94.22 | 95.52 | 
| Stanford Cars | CLIP | openai/clip-vit-large-patch14-336 | 16 | NaN | 1e-3 | 78.20 | 84.09 | 87.95 | 
| Food-101 | Image | swin_base_patch4_window7_224 | 16 | 1e-2 | 1e-3 | 73.66 | 73.64 | 76.18 | 
| Caltech101 | Image | swin_base_patch4_window7_224 | 16 | 1e-2 | 1e-3 | 96.75 | 96.75 | 97.16 | 
| DTD | Image | swin_base_patch4_window7_224 | 16 | 1e-2 | 1e-3 | 67.55 | 68.26 | 70.45 | 
| SetFit/sst5 | Text | sentence-transformers/paraphrase-mpnet-base-v2 | 16 | 1e-2 | 1e-3 | 38.42 | 38.42 | 39.23 | 
| SetFit/sst5 | Text | sentence-transformers/paraphrase-mpnet-base-v2 | 1 | 1e-2 | 1e-3 | 33.08 | 33.08 | 33.08 | 
| SetFit/sst5 | Text | sentence-transformers/paraphrase-mpnet-base-v2 | 64 | 1e-2 | 1e-3 | 45.61 | 46.02 | 48.19 | 
| SetFit/sst5 | Text | sentence-transformers/msmarco-MiniLM-L-12-v3 | 16 | 1e-2 | 1e-3 | 30.18 | 30.86 | 30.59 |
| SetFit/Emotion | Text | sentence-transformers/paraphrase-mpnet-base-v2 | 16 | 1e-2 | 1e-3 | 43.10 | 43.65 | 43.90 | 
| SetFit/subj | Text | sentence-transformers/paraphrase-mpnet-base-v2 | 16 | 1e-2 | 1e-3 | 90.50 | 90.55 | 90.75 | 
| SetFit/20_newsgroups | Text | sentence-transformers/paraphrase-mpnet-base-v2 | 16 | 1e-2 | 1e-3 | 54.14 | 57.36 | 58.90 | 
| SetFit/enron_spam | Text | sentence-transformers/paraphrase-mpnet-base-v2 | 16 | 1e-2 | 1e-3 | 91.35 | 91.70 | 92.85 | 
| SetFit/SentEval-CR | Text | sentence-transformers/paraphrase-mpnet-base-v2 | 16 | 1e-2 | 1e-3 | 88.31 | 88.58 | 89.24 | 


---

### Reference

[1] Tip-Adapter: Training-free CLIP-Adapter for Better Vision-Language Modeling. <https://arxiv.org/pdf/2111.03930.pdf>
