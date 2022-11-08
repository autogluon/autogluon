import os
import pandas as pd

from utils import (
    download,
    get_data_home_dir,
    get_repo_url,
    protected_zip_extraction,
    path_expander,
)


class AmazonReviewSentimentCrossLingualDataset:
    def __init__(
        self,
    ):
        sha1sum_id = "9c701aa6fc42ec3fe429bfe85a8dac4532ab9fcd"
        dataset = "amazon_review_sentiment_cross_lingual"
        file_name = f"{dataset}.zip"
        url = get_repo_url() + file_name
        save_path = os.path.join(get_data_home_dir(), file_name)
        self._path = os.path.join(get_data_home_dir(), dataset)
        download(
            url=url,
            path=save_path,
            sha1_hash=sha1sum_id,
        )
        protected_zip_extraction(
            save_path,
            sha1_hash=sha1sum_id,
            folder=get_data_home_dir(),
        )
        self._train_en_df = pd.read_csv(
            os.path.join(self._path, "en_train.tsv"),
            sep="\t",
            header=None,
            names=["label", "text"],
        ).sample(1000, random_state=123)

        self._test_en_df = pd.read_csv(
            os.path.join(self._path, "en_test.tsv"),
            sep="\t",
            header=None,
            names=["label", "text"],
        ).sample(200, random_state=123)

        self._train_en_df.reset_index(drop=True, inplace=True)
        self._test_en_df.reset_index(drop=True, inplace=True)

        print(f"train sample num: {len(self._train_en_df)}")
        print(f"test sample num: {len(self._test_en_df)}")

    @property
    def path(self):
        return self._path

    @property
    def label_columns(self):
        return ["label"]

    @property
    def train_df(self):
        return self._train_en_df

    @property
    def test_df(self):
        return self._test_en_df
