"""1. Text Classification - Data Preparation
==============================================

We show how to get custom data and prepare it in the format required by AutoGluon.

Step 1: Prepare Your Text Data (Optional)
----------------------------------------

AutoGluon currently supports .tsv and .txt formats. Additionally, GluonNLP Dataset formats (.json, .tsv) are also supported.

The output of this step would be (if using .tsv).

::

    data
    ├── train.tsv
    ├── val.tsv (optional)


where `data` is the data folder, `train.tsv` is a file that contains the training dataset and val.tsv contains the validation dataset.
The folder structure will be the same for .txt or .json format as well.

For .txt, the required data format in the file is : LABEL TEXT (Whitespace separated). eg :
    0 This movie is hilarious
    1 This movie is pathetic

For .tsv format, refer to `TSVDataset <https://gluon-nlp.mxnet.io/api/modules/data.html#gluonnlp.data.TSVDataset>'


We now show some examples for the supported datasets.

Example 1: Kaggle - Sentiment Analysis using Rotten Tomatoes Dataset
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Kaggle is a popular machine learning competition platform and contains lots of
datasets for different machine learning tasks including text classification.

Before downloading dataset, if you don't have Kaggle account,
please direct to `Kaggle <https://www.kaggle.com/>`__
to register one. Then please follow the Kaggle
`installation <https://github.com/Kaggle/kaggle-api/>`__ to install Kaggle API
for downloading the data.

For this example, we'll pick a competition on Kaggle titled : `Sentiment Analysis on Movie Reviews <https://www.kaggle.com/c/sentiment-analysis-on-movie-reviews/data>`.

This is a competition based on a dataset for movie reviews sourced from Rotten Tomatoes.

We then go to `Data` to download the dataset using the Kaggle API.

An example shell script to download the dataset to `~/data/sentiment_analysis/` can be found:

:download:`Download download_shopeeiet.sh<../../_static/script/download_sent_analysis_rotten_tomatoes.sh>`

Run it with

::

    sh download_sent_analysis_rotten_tomatoes.sh

Now we have the following structure under `~/data/sentiment_analysis/`:

::

    sentiment_analysis
    ├── train.tsv


Example 2: Kaggle - Quora Question Pairs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Similar to the above example, we picked up another NLP competition on Kaggle called : `Question Pairs <https://www.kaggle.com/c/quora-question-pairs>`.
This competition is based on detecting whether a pair of questions are similar semantically or not.

An example shell script to download the dataset to `~/data/quora_question_pairs` cane be found:

:download:`Download download_shopeeiet.sh<../../_static/script/download_quora_question_pairs.sh>`

Run it with

::

    sh download_quora_question_pairs.sh

Now we have the following structure under `~/data/quora_question_pairs/`:

::

    quora_question_pairs
    ├── train.csv


Step 2: Convert Data Source into Required Format
--------------------------------------------------

Training/Validation Split
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Training Set: The vast majority of your data should be in the training set.
This is the data your model "sees" during training:
it's used to learn the parameters of the model,
namely the weights of the connections between nodes of the neural network.

Validation Set: The validation set, sometimes also called the "dev" set,
is also used during the training process.
After the model learning framework incorporates training data
during each iteration of the training process,
it uses the model's performance on the validation set to tune the model's hyperparameters,
which are variables that specify the model's structure.

Manual Splitting: You can also split your dataset yourself.
Manually splitting your data is a good choice
when you want to exercise more control over the process
or if there are specific examples that you're sure you want
included in a certain part of your model training lifecycle.

The split of dataset into training and validation can generally vary. An example split can be as follows :

- Training Set: 90% of dataset.
- Validation Set: 10% of dataset.

We show an example below on how to convert data source obtained in Step 1
to Training/Validation split with the required format.

Example: Convert Kaggle - Sentiment Analysis using Rotten Tomatoes Dataset to Required Format
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We provide a script to convert Sentiment analysis dataset to the required format:

:download:`Download prepare_sent_analysis_data.py<../../_static/script/prepare_sent_analysis_data.py>`

After running with:

::

    python prepare_sent_analysis_data.py --data ~/data/sentiment_analysis/train.tsv --split 0.9

(`split` in the above command refers to the split of the data into training and validation datasets).

Thus, the resulting data is in the following format:

::

    sentiment_analysis
    ├── train.tsv
    ├── val.tsv

Similarly, we have provided another script to convert the Quora Question Pairs dataset to the required format :

:download:`Download prepare_quora_question_pairs_data.py<../../_static/script/prepare_quora_question_pairs_data.py>`

Which can be run with :
::

    python prepare_quora_question_pairs_data.py --data ~/data/quora_question_pairs/train.csv --split 0.9

The output of this script is similar to the one before.

Now you have a dataset ready to be used in AutoGluon.
"""
